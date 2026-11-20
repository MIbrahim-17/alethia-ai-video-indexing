"""Run run.py with Python networking blocked, to check the timed run works offline (D-025).

Every DNS lookup and socket connection made from Python is refused (OSError, as on a
machine with no network) and logged to stderr. At the end it prints how many attempts
were made. Unprivileged network namespaces are disabled on the dev server, so this is the
no-sudo way to test it. CUDA and TensorRT don't use the network themselves.

Usage:  python validation/offline_guard.py run.py --input ... --output ... [run.py options]
"""

import atexit
import runpy
import socket
import sys
import traceback

attempts = []


def _refuse(what):
    def f(*args, **kwargs):
        attempts.append((what, repr(args[:2])[:120]))
        stack = "".join(traceback.format_stack(limit=6)[:-1])
        print(f"OFFLINE-GUARD blocked {what}{args[:2]!r}\n{stack}", file=sys.stderr)
        raise OSError(101, "Network is unreachable (offline_guard)")
    return f


socket.getaddrinfo = _refuse("getaddrinfo")
socket.create_connection = _refuse("create_connection")
socket.socket.connect = _refuse("connect")
socket.socket.connect_ex = _refuse("connect_ex")
atexit.register(lambda: print(f"OFFLINE-GUARD network attempts: {len(attempts)} {attempts[:5]}", file=sys.stderr))

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
