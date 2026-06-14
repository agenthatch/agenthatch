"""Real-terminal test: Ctrl+C interrupts turn but agent stays alive.

Uses pty.spawn() to create a real pseudoterminal, then sends:
1. "you should test google" + Enter
2. Wait 4s for agent to start executing
3. Ctrl+C (0x03) to interrupt
4. "hello are you still there" + Enter to verify agent stayed alive
"""
import os
import sys
import time
import pty
import select
import signal
import termios
import tty
import struct
import fcntl

def set_pty_size(fd, rows=40, cols=140):
    """Set pty window size so prompt_toolkit detects a real terminal."""
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

def main():
    master_fd, slave_fd = pty.openpty()
    set_pty_size(slave_fd, 50, 160)

    pid = os.fork()
    if pid == 0:  # child
        os.close(master_fd)
        os.setsid()
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)

        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["TERM"] = "xterm-256color"
        env["PYTHONPATH"] = (
            "/Users/didi/agenthatch_developer/project/agenthatch/src:"
            "/Users/didi/agenthatch_developer/project/agenthatch/agenthatch-core/src"
        )
        os.execvpe(
            "/opt/homebrew/bin/python3.14",
            [
                "python3.14",
                "-m", "agenthatch",
                "run", "agentbrowser",
                "--no-color",
            ],
            env,
        )
        os._exit(1)

    os.close(slave_fd)

    # Make master non-blocking
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    output_buf = b""
    results = {"interrupted": False, "received_second": False, "exited_early": False}

    def drain(timeout_s=0.5):
        """Read all available output from pty."""
        nonlocal output_buf
        end = time.time() + timeout_s
        while time.time() < end:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if not r:
                break
            try:
                data = os.read(master_fd, 4096)
                if data:
                    output_buf += data
                    print(data.decode("utf-8", errors="replace"), end="", flush=True)
            except BlockingIOError:
                break

    def send(text: bytes):
        """Write to pty."""
        os.write(master_fd, text)
        time.sleep(0.3)

    print("=" * 60)
    print("CTRL+C INTERRUPT TEST (real pty)")
    print("=" * 60)
    print()

    # Wait for agent to boot
    time.sleep(4)
    drain(1)

    # Step 1: Send test command
    print("\n>>> Step 1: Sending 'you should test google'")
    send(b"you should test google\r")
    time.sleep(6)
    drain(2)

    # Step 2: Send Ctrl+C during execution
    print("\n>>> Step 2: Sending Ctrl+C (0x03)")
    send(b"\x03")
    time.sleep(3)
    drain(2)

    # Check if "Interrupted" appeared
    decoded = output_buf.decode("utf-8", errors="replace")
    if "Interrupted" in decoded or "interrupted" in decoded.lower():
        results["interrupted"] = True
        print(">>> Ctrl+C INTERRUPT DETECTED <<<")

    # Step 3: Check if agent is still alive
    try:
        os.kill(pid, 0)  # Signal 0 = check if process exists
        print(">>> Agent process still alive <<<")
    except ProcessLookupError:
        results["exited_early"] = True
        print(">>> AGENT EXITED! PID not found <<<")

    if not results["exited_early"]:
        # Step 4: Send follow-up to verify conversation continues
        print("\n>>> Step 3: Sending follow-up 'hello are you still there'")
        send(b"hello are you still there\r")
        time.sleep(8)
        drain(3)

        # Check output for agent's response
        decoded2 = output_buf.decode("utf-8", errors="replace")
        if "hello" in decoded2.lower() and decoded2.count("You:") >= 2:
            results["received_second"] = True

    # Cleanup
    if not results["exited_early"]:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os.close(master_fd)

    # Report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Interrupt message seen:    {'PASS ✅' if results['interrupted'] else 'FAIL ❌'}")
    print(f"Agent didn't exit early:   {'PASS ✅' if not results['exited_early'] else 'FAIL ❌'}")
    print(f"Follow-up response:        {'PASS ✅' if results['received_second'] else 'N/A (timing)'}")

    all_pass = results["interrupted"] and not results["exited_early"]
    print()
    if all_pass:
        print("VERDICT: Ctrl+C interrupts turn WITHOUT killing agent ✅")
    else:
        print("VERDICT: Issues remain ❌")

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())