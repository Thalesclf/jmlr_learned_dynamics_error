"""Minimal torch-free reader for PyTorch zip checkpoints (tensors -> NumPy)."""
import pickle, zipfile, collections
import numpy as np

_DT = {
    'DoubleStorage': np.float64, 'FloatStorage': np.float32, 'HalfStorage': np.float16,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
    'ComplexDoubleStorage': np.complex128, 'ComplexFloatStorage': np.complex64,
}

class _StorageType:
    def __init__(self, name): self.name = name

def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    arr, = storage
    itemsize = arr.itemsize
    if len(size) == 0:
        return arr[offset].copy()
    out = np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                          strides=tuple(s * itemsize for s in stride))
    return np.array(out)

def _rebuild_parameter(data, requires_grad, hooks, *a):
    return data

class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f); self.zf = zf; self.prefix = prefix; self.cache = {}
    def find_class(self, module, name):
        if module.startswith('torch'):
            if name == '_rebuild_tensor_v2': return _rebuild_tensor_v2
            if name == '_rebuild_parameter': return _rebuild_parameter
            if name.endswith('Storage'): return _StorageType(name)
            if name in ('device',): return lambda *a, **k: str(a)
            if name in ('float64','float32','complex128','complex64','int64'):
                return name
            raise pickle.UnpicklingError(f'unsupported torch global {module}.{name}')
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'builtins':
            import builtins; return getattr(builtins, name)
        if module == 'numpy.core.multiarray' or module == 'numpy._core.multiarray' or module.startswith('numpy'):
            import importlib; return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')
    def persistent_load(self, pid):
        typ, stype, key, loc, numel = pid
        if key not in self.cache:
            name = stype.name if isinstance(stype, _StorageType) else str(stype)
            raw = self.zf.read(f'{self.prefix}/data/{key}')
            self.cache[key] = np.frombuffer(raw, dtype=_DT[name]).copy()
        return (self.cache[key],)

def load(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith('data.pkl')][0]
    prefix = pkl[:-len('/data.pkl')]
    with zf.open(pkl) as f:
        return _Unpickler(f, zf, prefix).load()
