"""Isolated code sandbox with pluggable secure backends.

Generated code is NEVER executed on the host. A secure backend (Docker or
Bubblewrap) provides real isolation: read-only root filesystem, empty
writable work directory, network namespace disabled, no host env/secrets,
and CPU/memory/process/wall-time limits.

FAIL-CLOSED: when no secure backend is available, execution is refused with
status=unavailable — the host-subprocess backend is legacy and unsafe, and
is never selected for production paths.
"""
import functools
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

ALLOWED_LANGUAGES = ("python", "py")

# Counts actual secure-sandbox executions (benchmark provenance).
_SECURE_SANDBOX_RUNS = 0


def sandbox_run_count() -> int:
    """Total secure-sandbox executions since process start."""
    return _SECURE_SANDBOX_RUNS

# Pinned by digest for reproducibility — never resolved at request time.
DOCKER_IMAGE = "python:3.11-slim@sha256:94c50be2dc994b873b55bc123e95e6dbade08095b3dfd790f51c34de3f08cbb7"
_BACKEND_ORDER: list[str] = ["docker", "bubblewrap"]


@dataclass
class SandboxResult:
    status: str = "unavailable"  # ok | timeout | resource_error | startup_error | unavailable
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    latency_ms: float = 0.0
    network_isolated: bool = False
    isolation: str = "none"  # full | none
    backend: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.exit_code == 0


class SandboxBackend:
    """Interface for a secure sandbox backend."""
    name: str = "base"
    unsafe: bool = False

    def available(self) -> bool:
        return False

    def isolation_level(self) -> str:
        return "full" if self.available() else "none"

    def run(
        self,
        code: str,
        *,
        files: dict[str, str] | None = None,
        timeout_seconds: float = 5.0,
        max_memory_mb: int = 256,
        max_cpu_seconds: int = 10,
        max_processes: int = 16,
        allow_network: bool = False,
    ) -> SandboxResult:
        raise NotImplementedError


def _write_workdir(code: str, files: dict[str, str] | None):
    """Create a fresh private workdir with script.py + extra files."""
    workdir = tempfile.mkdtemp(prefix="nim-sandbox-")
    script_path = os.path.join(workdir, "script.py")
    with open(script_path, "w") as f:
        f.write(code)
    for name, content in (files or {}).items():
        safe = os.path.basename(name)
        with open(os.path.join(workdir, safe), "w") as f:
            f.write(content)
    return workdir, script_path


_SANDBOX_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "HOME": "/tmp",
    "TMPDIR": "/tmp",
}


def _setrlimit(resource_id: int, value: int) -> None:
    try:
        current = resource.getrlimit(resource_id)
        hard = current[1]
        if hard >= 0 and value > hard:
            value = hard
        resource.setrlimit(resource_id, (value, value))
    except (ValueError, OSError):
        pass  # cannot enforce here — degrade, do not fail


def _child_limits(max_memory_mb: int, max_cpu_seconds: int, max_processes: int) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    limit = max_memory_mb * 1024 * 1024
    _setrlimit(resource.RLIMIT_AS, limit)
    _setrlimit(resource.RLIMIT_CPU, max_cpu_seconds)
    _setrlimit(resource.RLIMIT_NPROC, max_processes)
    _setrlimit(resource.RLIMIT_NOFILE, 64)
    _setrlimit(resource.RLIMIT_FSIZE, 2 * 1024 * 1024)


def _classify(exit_code: int | None, stderr: str, status: str) -> str:
    if status != "ok":
        return status
    if exit_code is None:
        return "timeout"
    if exit_code == -24:  # SIGXCPU from RLIMIT_CPU
        return "resource_error"
    if "MemoryError" in stderr or "Cannot allocate memory" in stderr:
        return "resource_error"
    # a plain nonzero exit (e.g. raised exception) stays "ok" with exit_code
    # set; callers must check SandboxResult.ok (status AND exit_code == 0)
    return "ok"


class HostSubprocessBackend(SandboxBackend):
    """LEGACY host subprocess backend — NOT isolated (same host, no mount or
    network namespace). unsafe=True: never selected for production paths;
    kept only for interface/limit unit tests."""

    name = "host_subprocess"
    unsafe = True

    def available(self) -> bool:
        return True

    def isolation_level(self) -> str:
        return "none"

    def run(self, code, *, files=None, timeout_seconds=5.0, max_memory_mb=256,
            max_cpu_seconds=10, max_processes=16, allow_network=False) -> SandboxResult:
        if not code.strip():
            return SandboxResult(status="startup_error", error="empty code", backend=self.name)
        t0 = time.monotonic()
        workdir, script_path = _write_workdir(code, files)
        env = dict(_SANDBOX_ENV)
        env["HOME"] = workdir
        env["TMPDIR"] = workdir

        def _preexec() -> None:
            _child_limits(max_memory_mb, max_cpu_seconds, max_processes)

        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", script_path],
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # preexec_fn is the only portable way to enforce RLIMIT_* in
                # the child; this legacy backend is single-threaded by design.
                preexec_fn=_preexec,  # noqa: PLW1509
                close_fds=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                status = "ok"
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                stdout, stderr = proc.communicate()
                status = "timeout"
                exit_code = None
        except Exception as e:
            return SandboxResult(
                status="startup_error", error=str(e), backend=self.name,
                latency_ms=(time.monotonic() - t0) * 1000, isolation="none",
            )

        err_text = stderr.decode(errors="replace")
        return SandboxResult(
            status=_classify(exit_code, err_text, status),
            stdout=stdout.decode(errors="replace"),
            stderr=err_text,
            exit_code=exit_code,
            latency_ms=(time.monotonic() - t0) * 1000,
            network_isolated=False,
            isolation="none",
            backend=self.name,
        )


class BubblewrapBackend(SandboxBackend):
    """Bubblewrap with a MINIMAL filesystem namespace: only the Python
    runtime directories and the isolated workdir are visible. The host root
    is never mounted — /home, the repository, /etc and secrets stay
    invisible."""

    name = "bubblewrap"

    @staticmethod
    def _runtime_binds() -> list[str]:
        """Binds for the python runtime only: the interpreter's own prefix,
        the system loader/libs and ld.so.cache. Nothing else."""
        real = os.path.realpath(sys.executable)
        prefix = os.path.dirname(os.path.dirname(real))  # .../cpython-3.11.15/
        binds: list[str] = []
        for src in (prefix, "/usr", "/lib", "/lib64", "/bin", "/sbin"):
            if os.path.isdir(src):
                binds.extend(["--ro-bind", src, src])
        for src in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/nsswitch.conf"):
            if os.path.isfile(src):
                binds.extend(["--ro-bind", src, src])
        return binds

    def available(self) -> bool:
        if shutil.which("bwrap") is None:
            return False
        try:
            # Probe a REAL python run inside the minimal namespace, not `true`
            with tempfile.TemporaryDirectory(prefix="nim-bwrap-probe-") as td:
                probe = subprocess.run(
                    ["bwrap", *self._runtime_binds(),
                     "--bind", td, td,
                     "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                     "--unshare-net", "--unshare-pid",
                     "--", sys.executable, "-I", "-S", "-c", "print(1)"],
                    cwd=td, capture_output=True, timeout=15, check=False,
                )
                return probe.returncode == 0 and probe.stdout.strip() == b"1"
        except Exception:
            return False

    def run(self, code, *, files=None, timeout_seconds=5.0, max_memory_mb=256,
            max_cpu_seconds=10, max_processes=16, allow_network=False) -> SandboxResult:
        if not code.strip():
            return SandboxResult(status="startup_error", error="empty code", backend=self.name)
        t0 = time.monotonic()
        workdir, script_path = _write_workdir(code, files)
        env = dict(_SANDBOX_ENV)
        env["HOME"] = workdir
        env["TMPDIR"] = workdir

        cmd = [
            "bwrap",
            *self._runtime_binds(),      # minimal rootfs: python runtime + loader
            "--bind", workdir, workdir,  # empty writable work directory
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--unshare-net",             # network namespace disabled
            "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--die-with-parent",
            "--new-session",
            "--",
            sys.executable, "-I", "-S", script_path,
        ]

        def _preexec() -> None:
            _child_limits(max_memory_mb, max_cpu_seconds, max_processes)

        try:
            proc = subprocess.Popen(
                cmd, cwd=workdir, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=_preexec,  # noqa: PLW1509
                close_fds=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                status, exit_code = "ok", proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                stdout, stderr = proc.communicate()
                status, exit_code = "timeout", None
        except Exception as e:
            return SandboxResult(
                status="startup_error", error=str(e), backend=self.name,
                latency_ms=(time.monotonic() - t0) * 1000, isolation="full",
            )

        err_text = stderr.decode(errors="replace")
        return SandboxResult(
            status=_classify(exit_code, err_text, status),
            stdout=stdout.decode(errors="replace"),
            stderr=err_text,
            exit_code=exit_code,
            latency_ms=(time.monotonic() - t0) * 1000,
            network_isolated=True,
            isolation="full",
            backend=self.name,
        )


@functools.lru_cache(maxsize=1)
def _docker_available() -> bool:
    """Docker is available only when the daemon runs AND the PINNED image is
    already present. The sandbox never pulls images during a request — use
    scripts/sandbox_setup.sh as the explicit preflight."""
    try:
        info = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
        if info.returncode != 0:
            return False
        inspect = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE], capture_output=True, timeout=10, check=False
        )
        return inspect.returncode == 0
    except Exception:
        return False


class DockerBackend(SandboxBackend):
    """Docker: --network none, --read-only root, tmpfs /tmp, memory/cpu/pids
    limits, run as nobody, host paths and env never visible."""

    name = "docker"

    def available(self) -> bool:
        return _docker_available()

    def run(self, code, *, files=None, timeout_seconds=5.0, max_memory_mb=256,
            max_cpu_seconds=10, max_processes=16, allow_network=False) -> SandboxResult:
        if not code.strip():
            return SandboxResult(status="startup_error", error="empty code", backend=self.name)
        t0 = time.monotonic()
        workdir, _ = _write_workdir(code, files)
        os.chmod(workdir, 0o777)  # writable by the container's nobody user
        for root, _, fnames in os.walk(workdir):
            for fn in fnames:
                os.chmod(os.path.join(root, fn), 0o644)

        name = f"nim-sbx-{os.getpid()}-{int(time.monotonic() * 1000) % 100000}"
        cmd = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",                 # network namespace disabled
            "--read-only",                        # read-only root filesystem
            "--tmpfs", "/tmp:rw,size=16m",
            "--memory", f"{max_memory_mb}m",
            "--memory-swap", f"{max_memory_mb}m",
            "--cpus", "1",
            "--pids-limit", str(max_processes),
            "--stop-timeout", "3",
            "--user", "65534:65534",              # nobody — no host privileges
            "-v", f"{workdir}:/work",
            "-w", "/work",
            "-i",
            DOCKER_IMAGE,
            "python3", "-I", "-S", "/work/script.py",
        ]
        env = dict(_SANDBOX_ENV)
        try:
            proc = subprocess.Popen(
                cmd, cwd=workdir, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                close_fds=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                status, exit_code = "ok", proc.returncode
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=5, check=False)
                stdout, stderr = proc.communicate()
                status, exit_code = "timeout", None
        except Exception as e:
            return SandboxResult(
                status="startup_error", error=str(e), backend=self.name,
                latency_ms=(time.monotonic() - t0) * 1000, isolation="full",
            )

        err_text = stderr.decode(errors="replace")
        return SandboxResult(
            status=_classify(exit_code, err_text, status),
            stdout=stdout.decode(errors="replace"),
            stderr=err_text,
            exit_code=exit_code,
            latency_ms=(time.monotonic() - t0) * 1000,
            network_isolated=True,
            isolation="full",
            backend=self.name,
        )


_SECURE_BACKENDS = [DockerBackend(), BubblewrapBackend()]  # docker preferred


@functools.lru_cache(maxsize=1)
def select_secure_backend() -> SandboxBackend | None:
    """Pick the first available secure backend. Docker is preferred (strongest
    isolation); Bubblewrap runs with a minimal rootfs. Never returns an
    unsafe one."""
    for b in _SECURE_BACKENDS:
        if b.available():
            return b
    return None


def run_secure_sandbox(
    code: str,
    *,
    language: str = "python",
    files: dict[str, str] | None = None,
    timeout_seconds: float = 5.0,
    max_memory_mb: int = 256,
    max_cpu_seconds: int = 10,
    max_processes: int = 16,
    allow_network: bool = False,
) -> SandboxResult:
    """Run untrusted code through a SECURE backend.

    Fails closed: without docker/bwrap, execution is refused (status
    unavailable) — the host subprocess fallback is never used.
    """
    if language not in ALLOWED_LANGUAGES:
        return SandboxResult(status="startup_error", error=f"language '{language}' not allowed")
    if not code.strip():
        return SandboxResult(status="startup_error", error="empty code")
    backend = select_secure_backend()
    if backend is None:
        return SandboxResult(
            status="unavailable",
            error="no secure sandbox backend available (docker or bwrap) — refusing host subprocess execution",
            isolation="none",
        )
    global _SECURE_SANDBOX_RUNS
    _SECURE_SANDBOX_RUNS += 1
    return backend.run(
        code,
        files=files,
        timeout_seconds=timeout_seconds,
        max_memory_mb=max_memory_mb,
        max_cpu_seconds=max_cpu_seconds,
        max_processes=max_processes,
        allow_network=allow_network,
    )


if __name__ == "__main__":
    """Preflight: report backend status. Run via scripts/sandbox_setup.sh."""
    backend = select_secure_backend()
    if backend is None:
        print("sandbox backend: NONE — secure execution fails closed")
        print("docker:", _docker_available(), "| bubblewrap:", BubblewrapBackend().available())
    else:
        print(f"sandbox backend: {backend.name} (isolation={backend.isolation_level()})")
        r = run_secure_sandbox("print('preflight-ok')")
        print(f"probe: status={r.status} stdout={r.stdout.strip()!r}")
        if not r.ok:
            raise SystemExit(1)
