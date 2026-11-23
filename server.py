import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_DIR = Path(__file__).resolve().parent
STARTUP_TIMEOUT = 120
SERVICES = (
    ("Website", ["-m", "http.server", "5500", "--bind", "127.0.0.1"], "http://127.0.0.1:5500/"),
    ("Scam AI", ["scam.py"], "http://127.0.0.1:5000/"),
    ("Deepfake AI", ["deepfake.py"], "http://127.0.0.1:5001/api/deepfake/health"),
    ("User API", ["users.py"], "http://127.0.0.1:5002/api/health"),
)


def start_server(args):
    return subprocess.Popen(
        [sys.executable, "-u", *args],
        cwd=PROJECT_DIR,
    )


def wait_until_ready(processes):
    pending = {name: url for name, _, url in SERVICES}
    deadline = time.monotonic() + STARTUP_TIMEOUT

    while pending and time.monotonic() < deadline:
        for name, process in processes.items():
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{name} exited during startup with code {exit_code}."
                )

        for name, url in list(pending.items()):
            try:
                with urlopen(url, timeout=1):
                    print(f"{name} is ready: {url}", flush=True)
                del pending[name]
            except (OSError, URLError):
                pass

        if pending:
            time.sleep(0.5)

    if pending:
        waiting_for = ", ".join(pending)
        raise TimeoutError(
            f"Timed out waiting for these services to start: {waiting_for}."
        )


def stop_servers(processes):
    for process in processes.values():
        if process.poll() is None:
            process.terminate()

    for process in processes.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    processes = {}
    exit_code = 0

    try:
        for name, args, _ in SERVICES:
            print(f"Starting {name}...", flush=True)
            processes[name] = start_server(args)

        wait_until_ready(processes)
        print("\nAll OffGrid services are ready.", flush=True)
        print("Press Ctrl+C to stop them.", flush=True)

        while True:
            for name, process in processes.items():
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"{name} stopped unexpectedly with code {exit_code}."
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping OffGrid services...", flush=True)
    except (OSError, RuntimeError, TimeoutError) as error:
        print(f"\nCould not start OffGrid services: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        stop_servers(processes)
        print("OffGrid services stopped.", flush=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
