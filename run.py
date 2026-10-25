import os
import signal
import subprocess
import sys
import time
import requests

PYTHON_EXEC = sys.executable
PORT = 8000


def kill_process_on_port(port: int):
    try:
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        for pid_str in pids:
            if pid_str:
                pid = int(pid_str)
                print(f"Port {port} is occupied by PID {pid}. Terminating process...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
    except FileNotFoundError:
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            time.sleep(0.5)
        except FileNotFoundError:
            pass
    except Exception as e:
        print(f"Warning: Could not check/kill process on port {port}: {e}")


def main():
    kill_process_on_port(PORT)

    print("Starting FastAPI server...")
    server_proc = subprocess.Popen(
        [PYTHON_EXEC, "-m", "uvicorn", "main_server:app", "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    while True:
        try:
            res = requests.get(f"http://127.0.0.1:{PORT}/")
            if res.status_code == 200:
                print("Server is ready!\n")
                break
        except requests.exceptions.ConnectionError:
            if server_proc.poll() is not None:
                print(f"Error: Server failed to start on port {PORT}.")
                sys.exit(1)
            time.sleep(0.5)

    try:
        subprocess.run([PYTHON_EXEC, "term_client.py"])
    finally:
        print("\nShutting down server...")
        server_proc.terminate()
        server_proc.wait()


if __name__ == "__main__":
    main()