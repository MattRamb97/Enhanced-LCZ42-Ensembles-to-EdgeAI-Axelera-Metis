import h5py
import numpy as np

def _np_string(value: str):
    """Return a numpy bytes scalar compatible with NumPy 1.x and 2.x."""
    string_ctor = getattr(np, "string_", None)
    if string_ctor is None:
        string_ctor = np.bytes_
    return string_ctor(value)

def save_h5_results(
    h5_path,
    name,
    classes,
    top1,
    cm,
    y_true,
    y_pred,
    history=None,
    extra=None,
    probs=None,
):
    if history is None:
        history = {}
    if extra is None:
        extra = {}

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("name", data=_np_string(name))
        f.create_dataset("top1", data=np.float32(top1))
        f.create_dataset("confusionMat", data=cm.astype(np.int32))
        f.create_dataset("yTrue", data=y_true.astype(np.int32))
        f.create_dataset("yPred", data=y_pred.astype(np.int32))
        if probs is not None:
            f.create_dataset("probs", data=np.asarray(probs, dtype=np.float32))

        # class names
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("classes", data=np.array(classes, dtype=dt))

        # training history
        hist_grp = f.create_group("history")
        for key, vals in history.items():
            hist_grp.create_dataset(key, data=np.array(vals, dtype=np.float32))

        # metadata / extra
        extra_grp = f.create_group("extra")
        for k, v in extra.items():
            if isinstance(v, str):
                # JSON strings are stored cleanly
                extra_grp.create_dataset(k, data=_np_string(v))
            else:
                extra_grp.create_dataset(k, data=np.array(v))

    print(f"Results saved → {h5_path}")
