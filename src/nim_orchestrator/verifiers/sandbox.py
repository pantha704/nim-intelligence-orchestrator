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

DOCKER_IMAGE = "python:3.11-slim"
_BACKEND_ORDER: list[str] = ["bubblewrap", "docker"]


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
    """Bubblewrap: read-only root, writable workdir, network namespace off."""

    name = "bubblewrap"

    def available(self) -> bool:
        if shutil.which("bwrap") is None:
            return False
        try:
            probe = subprocess.run(
                ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                 "--tmpfs", "/tmp", "--unshare-net", "--unshare-pid", "true"],
                capture_output=True, timeout=10, check=False,
            )
            return probe.returncode == 0
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
            "--ro-bind", "/", "/",          # read-only root filesystem
            "--bind", workdir, workdir,     # empty writable work directory
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--unshare-net",                # network namespace disabled
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
    try:
        info = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
        if info.returncode != 0:
            return False
        inspect = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE], capture_output=True, timeout=10, check=False
        )
        if inspect.returncode == 0:
            return True
        pull = subprocess.run(["docker", "pull", DOCKER_IMAGE], capture_output=True, timeout=120, check=False)
        return pull.returncode == 0
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


_SECURE_BACKENDS = [BubblewrapBackend(), DockerBackend()]


@functools.lru_cache(maxsize=1)
def select_secure_backend() -> SandboxBackend | None:
    """Pick the first available secure backend. Never returns an unsafe one."""
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
    return backend.run(
        code,
        files=files,
        timeout_seconds=timeout_seconds,
        max_memory_mb=max_memory_mb,
        max_cpu_seconds=max_cpu_seconds,
        max_processes=max_processes,
        allow_network=allow_network,
    )
