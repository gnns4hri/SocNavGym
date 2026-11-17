"""The environment's side of the render server (server.py): starts it as a process of its own and exchanges messages.
The server needs a python that has Blender's `bpy` module. It is looked for in this order: the `python` argument (config:
vision.renderer.python), $SOCNAV_VISION_PYTHON, <repository>/.venv_render/bin/python, the running python."""
import os, sys, subprocess, tempfile, atexit
from multiprocessing.connection import Listener
from . import ipc

HERE = os.path.dirname(os.path.abspath(__file__))


def renderer_python(python=None):
    for p in (python, os.environ.get("SOCNAV_VISION_PYTHON"), os.path.join(os.path.dirname(os.path.dirname(HERE)), ".venv_render", "bin", "python")):
        if p and os.path.exists(os.path.expanduser(p)): return os.path.expanduser(p)
    return sys.executable


_checked = {}


def renderer_missing(python=None):
    """None when the renderer can be started; otherwise what is missing (a sentence). Looks, without starting Blender, for the
    renderer's python, the `bpy` module in it and the built 3D models."""
    py = renderer_python(python)
    if py not in _checked:
        why = None
        try:
            if py == sys.executable:
                import importlib.util; found = importlib.util.find_spec("bpy") is not None
            else: found = subprocess.run([py, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('bpy') else 3)"], capture_output=True, timeout=60).returncode == 0
        except Exception: found = False
        if not found: why = f"Blender's `bpy` module is not installed in the renderer's python ({py})"
        elif not any(f.endswith(".blend") for f in (os.listdir(os.path.join(HERE, "assets")) if os.path.isdir(os.path.join(HERE, "assets")) else [])): why = f"the 3D models are not built ({os.path.join(HERE, 'assets')} is empty)"
        _checked[py] = why
    return _checked[py]


class RenderClient:
    def __init__(self, python=None, log=None, timeout=300.0, **settings):
        self.dir = tempfile.mkdtemp(prefix="socnav_vision_"); addr = os.path.join(self.dir, "sock"); self.timeout = timeout
        lis = Listener(addr, family="AF_UNIX"); lis._listener._socket.settimeout(timeout)
        self.log = open(log, "w") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen([renderer_python(python), os.path.join(HERE, "server.py"), addr], stdout=self.log, stderr=subprocess.STDOUT)
        try: self.conn = lis.accept()
        except Exception as e:
            self.proc.kill(); raise RuntimeError("the render server did not start (is Blender's `bpy` module installed in the renderer's python? "
                                                 f"tried {renderer_python(python)}; see socnavgym/vision/README.md)") from e
        finally: lis.close()
        atexit.register(self.close); self.info = self.call("init", **settings)

    def call(self, op, **kw):
        kw["op"] = op
        try:
            self.conn.send_bytes(ipc.pack(kw))
            if not self.conn.poll(self.timeout): raise TimeoutError(f"the render server did not answer '{op}' in {self.timeout:.0f} s")
            r = ipc.unpack(self.conn.recv_bytes())
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            code = self.proc.poll() if self.proc is not None else None
            raise RuntimeError(f"the render server stopped while answering '{op}' (exit code {code}); set vision.renderer.log to a file name to keep its output") from e
        if not r.get("ok"): raise RuntimeError("render server error:\n" + r.get("error", "?"))
        return r

    def close(self):
        if getattr(self, "proc", None) is None: return
        try:
            if self.proc.poll() is None: self.conn.send_bytes(ipc.pack({"op": "close"})); self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        pid = self.proc.pid; self.proc = None
        try:
            f = f"/dev/shm/socnav_vision_{pid}.exr"               # the server's scratch picture, if it could not remove it itself
            if os.path.exists(f): os.remove(f)
            self.conn.close()
            if self.log is not subprocess.DEVNULL: self.log.close()
            for f in os.listdir(self.dir): os.remove(os.path.join(self.dir, f))
            os.rmdir(self.dir)
        except OSError: pass
