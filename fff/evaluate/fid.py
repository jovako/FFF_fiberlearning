import numpy as np
import torch
from pathlib import Path
import sys
import io
from contextlib import redirect_stdout
import tensorflow.compat.v1 as tf
sys.path.append("/home/hd/hd_hd/hd_gu452/oc-guidance/")
from .guided_diffusion_evaluator import main



def save_torch_images_as_npz(
    images: torch.Tensor,
    path: str,
):
    """
    Save a torch tensor as an NPZ file compatible with OpenAI FID code.

    images: (N, 3, H, W), values in [-1, 1]
    """
    assert images.ndim == 4 and images.shape[1] == 3
    images = images.detach().cpu()

    # [-1, 1] -> [0, 255]
    images = (images + 1) * 127.5
    images = images.clamp(0, 255).byte()

    # NCHW -> NHWC
    images = images.permute(0, 2, 3, 1).numpy()

    np.savez(path, images)

def compute_fid_openai_tf(
    samples_1: torch.Tensor,
    samples_2: torch.Tensor,
    tmp_dir: str = "./fid_tmp",
    delete_tmp_dir = False,
):
    """
    Compute FID using OpenAI's original TensorFlow implementation.

    Returns:
        fid (float)
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ref_path = tmp_dir / "ref.npz"
    sample_path = tmp_dir / "sample.npz"

    save_torch_images_as_npz(samples_1, ref_path)
    save_torch_images_as_npz(samples_2, sample_path)

    # --- Run OpenAI FID code ---
    # We capture stdout and parse the FID value.
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        # Reset TF graph to avoid collisions
        tf.reset_default_graph()

        # Simulate command-line arguments
        sys.argv = [
            "guided_diffusion_evaluator.py",
            str(ref_path),
            str(sample_path),
        ]

        main()  # <-- this is the `main()` from the OpenAI script

    output = buffer.getvalue()
    if delete_tmp_dir:
        os.rmdir(tmp_dir)
    # Parse FID
    fid = None
    sfid = None
    IS = None
    precision = None
    recall = None
    print(output)
    for line in output.splitlines():
        if line.startswith("Inception Score:"):
            IS = float(line.split("Inception Score:")[1].strip())
        if line.startswith("FID:"):
            fid = float(line.split("FID:")[1].strip())
        if line.startswith("sFID:"):
            sfid = float(line.split("sFID:")[1].strip())
        if line.startswith("Precision:"):
            precision = float(line.split("Precision:")[1].strip())
        if line.startswith("Recall:"):
            recall = float(line.split("Recall:")[1].strip())
            return IS, fid, sfid, precision, recall
    raise RuntimeError("Recall value not found in output:\n" + output)
