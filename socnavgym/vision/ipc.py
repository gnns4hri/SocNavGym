"""Messages between the environment's process and the render server (another python: Blender's module needs its own).
A message is a dict of plain values (numbers, strings, lists, dicts) in which numpy arrays may appear anywhere; it travels as
one byte string: a JSON header with the arrays replaced by references, followed by the arrays' raw bytes.
Only the standard library and numpy: this file is imported by both sides."""
import json, struct
import numpy as np

# semantic classes of the ground-truth segmentation (images and lidar)
CLASSES = {"background": 0, "floor": 1, "wall": 2, "table": 3, "chair": 4, "plant": 5, "laptop": 6, "human": 7, "robot": 8, "ceiling": 9}


def pack(msg):
    blobs = []
    def enc(v):
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v); blobs.append(a.tobytes()); return {"__nd__": len(blobs) - 1, "dtype": str(a.dtype), "shape": list(a.shape)}
        if isinstance(v, dict): return {str(k): enc(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [enc(x) for x in v]
        if isinstance(v, np.generic): return v.item()
        return v
    head = json.dumps(enc(msg)).encode()
    return b"".join([struct.pack("<II", len(head), len(blobs)), head] + [struct.pack("<Q", len(b)) + b for b in blobs])


def unpack(data):
    nh, nb = struct.unpack_from("<II", data, 0); p = 8; head = json.loads(data[p:p + nh].decode()); p += nh; blobs = []
    for _ in range(nb):
        (n,) = struct.unpack_from("<Q", data, p); p += 8; blobs.append(data[p:p + n]); p += n
    def dec(v):
        if isinstance(v, dict):
            if "__nd__" in v: return np.frombuffer(blobs[v["__nd__"]], dtype=v["dtype"]).reshape(v["shape"]).copy()
            return {k: dec(x) for k, x in v.items()}
        if isinstance(v, list): return [dec(x) for x in v]
        return v
    return dec(head)
