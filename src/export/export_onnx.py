"""Export a PyTorch model to ONNX (dynamic batch) and verify against PyTorch."""
import numpy as np, torch, onnxruntime as ort


def export_and_verify(model, path, example, input_names=("input",), output_names=("output",),
                      opset=17, atol=1e-4):
    model = model.eval().cpu()
    ex = example if isinstance(example, tuple) else (example,)
    axes = {n: {0: "batch"} for n in (*input_names, *output_names)}
    try:
        torch.onnx.export(model, ex, path, input_names=list(input_names), output_names=list(output_names),
                          dynamic_axes=axes, opset_version=opset, dynamo=False)
    except TypeError:                                  # older torch without the dynamo flag
        torch.onnx.export(model, ex, path, input_names=list(input_names), output_names=list(output_names),
                          dynamic_axes=axes, opset_version=opset)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    report = {}
    for bs in (1, 4, 16):                              # test several batch sizes
        batch = tuple(torch.cat([e] * bs)[:bs] if e.shape[0] != bs else e for e in ex)
        batch = tuple(torch.rand(bs, *e.shape[1:], dtype=e.dtype) if e.dtype.is_floating_point else e[:1].repeat(bs, *[1]*(e.ndim-1)) for e in ex)
        with torch.no_grad():
            ref = model(*batch)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        out = sess.run(None, {n: b.numpy() for n, b in zip(input_names, batch)})
        report[bs] = max(float(np.abs(r.numpy() - o).max()) for r, o in zip(ref, out))
    ok = all(v < atol for v in report.values())
    return ok, report
