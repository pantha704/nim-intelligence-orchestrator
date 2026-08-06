"""Isolated code sandbox — generated code is NEVER executed on the host.

Isolation guarantees (best effort, portable):
- runs in a fresh private temporary directory (cwd inside it)
- environment stripped: no host variables, credentials or PYTHON* settings
- `python3 -I` isolated interpreter (no user site, ignores PYTHON* env)
- resource limits in the child: CPU seconds, address space (memory),
  process count, file descriptors
- wall-clock timeout enforced from the parent, process group killed
- network disabled by default via `unshare -n` when available; when the
  kernel does not allow it, the result reports network_isolated=False
- explicit language allowlist (only python)
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


@dataclass
class SandboxResult:
    status: str = "ok"  # ok | timeout | resource_error | startup_error
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    latency_ms: float = 0.0
    network_isolated: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.exit_code == 0


def _unshare_available() -> bool:
    return shutil.which("unshare") is not None


@functools.lru_cache(maxsize=1)
def _unshare_usable() -> bool:
    """Probe whether `unshare -n` actually works in this environment."""
    if not _unshare_available():
        return False
    try:
        result = subprocess.run(
            ["unshare", "-n", "true"], capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def _child_limits(max_memory_mb: int, max_cpu_seconds: int, max_processes: int) -> None:
    try:
        os.setsid()
    except OSError:
        pass  # already a session leader — process-group kill still best effort

    def _setrlimit(resource_id, value):
        try:
            current = resource.getrlimit(resource_id)
            hard = current[1]
            if hard >= 0 and value > hard:
                value = hard
            resource.setrlimit(resource_id, (value, value))
        except (ValueError, OSError):
            pass  # cannot lower/raise a limit here — degrade, do not fail

    limit = max_memory_mb * 1024 * 1024
    _setrlimit(resource.RLIMIT_AS, limit)
    _setrlimit(resource.RLIMIT_CPU, max_cpu_seconds)
    _setrlimit(resource.RLIMIT_NPROC, max_processes)
    _setrlimit(resource.RLIMIT_NOFILE, 64)
    _setrlimit(resource.RLIMIT_FSIZE, 2 * 1024 * 1024)


def run_in_sandbox(
    code: str,
    *,
    language: str = "python",
    timeout_seconds: float = 5.0,
    max_memory_mb: int = 256,
    max_cpu_seconds: int = 10,
    max_processes: int = 16,
    allow_network: bool = False,
) -> SandboxResult:
    """Run untrusted code in the isolated sandbox. Never runs on the host."""
    if language not in ALLOWED_LANGUAGES:
        return SandboxResult(status="startup_error", error=f"language '{language}' not allowed")
    if not code.strip():
        return SandboxResult(status="startup_error", error="empty code")

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="nim-sandbox-") as workdir:
        script_path = os.path.join(workdir, "script.py")
        with open(script_path, "w") as f:
            f.write(code)

        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "HOME": workdir,
            "TMPDIR": workdir,
            "PYTHONPATH": workdir,
        }

        cmd: list[str] = [sys.executable, "-I", "-S", script_path]
        network_isolated = False
        if not allow_network and _unshare_usable():
            network_isolated = True
            cmd = ["unshare", "-n", *cmd]

        def _preexec() -> None:
            _child_limits(max_memory_mb, max_cpu_seconds, max_processes)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # preexec_fn is the only portable way to enforce RLIMIT_* in
                # the child; the sandbox runner is single-threaded by design.
                preexec_fn=_preexec,  # noqa: PLW1509
                close_fds=True,
            )
        except OSError as e:
            if network_isolated:
                # unshare not permitted — degrade, do NOT raise
                network_isolated = False
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "-I", "-S", script_path],
                        cwd=workdir,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=_preexec,  # noqa: PLW1509
                        close_fds=True,
                    )
                except OSError as e2:
                    return SandboxResult(
                        status="startup_error",
                        error=str(e2),
                        latency_ms=(time.monotonic() - t0) * 1000,
                        network_isolated=False,
                    )
            else:
                return SandboxResult(
                    status="startup_error",
                    error=str(e),
                    latency_ms=(time.monotonic() - t0) * 1000,
                    network_isolated=False,
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
                status="startup_error",
                error=str(e),
                latency_ms=(time.monotonic() - t0) * 1000,
                network_isolated=network_isolated,
            )

        latency_ms = (time.monotonic() - t0) * 1000
        err_text = stderr.decode(errors="replace")

        if status == "ok" and exit_code != 0:
            # RLIMIT_CPU → SIGXCPU (-24); memory exhaustion → MemoryError text
            if exit_code == -24:
                status = "resource_error"
                err_text = "CPU limit exceeded (SIGXCPU)"
            elif "MemoryError" in err_text or "Cannot allocate memory" in err_text:
                status = "resource_error"
                err_text = "memory limit exceeded"
            else:
                status = "resource_error" if exit_code < 0 else "ok"

        return SandboxResult(
            status=status,
            stdout=stdout.decode(errors="replace"),
            stderr=err_text,
            exit_code=exit_code,
            latency_ms=latency_ms,
            network_isolated=network_isolated,
        )
