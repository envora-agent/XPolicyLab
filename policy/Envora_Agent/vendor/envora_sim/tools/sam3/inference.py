"""SAM 3 GPU worker: load once, then block until an IPC request arrives.

Run with the project's Python environment; see this folder's README.
The socket protocol uses one JSON line each way. RGB and mask files stay local.
The original one-shot CLI and stdin ``--serve`` protocol remain available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).with_name("checkpoints") / "sam3.pt",
    )
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.2)
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--socket", type=Path, help="listen on this Unix socket")
    transport.add_argument("--serve", action="store_true", help="read stdin requests")
    arguments = parser.parse_args()
    if (
        not arguments.serve
        and arguments.socket is None
        and None
        in (
            arguments.rgb,
            arguments.output,
            arguments.prompts,
        )
    ):
        parser.error("--rgb, --output and --prompts are required without a server")

    if not arguments.checkpoint.is_file() or arguments.checkpoint.stat().st_size < 1024:
        from huggingface_hub import hf_hub_download

        print(
            "SAM 3: downloading pinned weights from squarefk/sam3-mirror",
            file=sys.stderr,
        )
        with redirect_stdout(sys.stderr):
            downloaded = hf_hub_download(
                repo_id="squarefk/sam3-mirror",
                filename="sam3.pt",
                revision="b675df84f9418772b352b36357d23c1dcba3e767",
                token=False,
            )
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete, verified file; interrupted downloads stay in HF's cache.
        with tempfile.NamedTemporaryFile(
            dir=arguments.checkpoint.parent, suffix=".pt", delete=False
        ) as destination:
            temporary = Path(destination.name)
            try:
                digest = hashlib.sha256()
                with open(downloaded, "rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        destination.write(chunk)
                if digest.hexdigest() != (
                    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
                ):
                    raise ValueError("SAM 3 mirror checkpoint SHA-256 mismatch")
                destination.close()
                temporary.replace(arguments.checkpoint)
            finally:
                temporary.unlink(missing_ok=True)

    import numpy as np
    import torch
    from PIL import Image

    if arguments.sam3_repo is not None:
        sys.path.insert(0, str(arguments.sam3_repo))
    # All library diagnostics go to stderr; stdout is the IPC startup/protocol pipe.
    started = time.perf_counter()
    with redirect_stdout(sys.stderr), torch.inference_mode():
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.device("cuda"):
                model = build_sam3_image_model(
                    device="cuda",
                    eval_mode=True,
                    checkpoint_path=str(arguments.checkpoint),
                    load_from_HF=False,  # Never fall back to the gated upstream repository.
                    enable_segmentation=True,
                    enable_inst_interactivity=False,
                    compile=False,
                )
            processor = Sam3Processor(
                model, device="cuda", confidence_threshold=arguments.threshold
            )
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    def infer(request: dict) -> dict:
        """Execute one protocol request; intermediates are released on return."""
        nonlocal load_seconds
        operation = request.get("operation", "segment")
        if operation not in ("segment", "warmup"):
            raise ValueError(f"unknown operation: {operation}")
        output_folder = None
        if operation == "warmup":
            image = Image.new("RGB", (request["width"], request["height"]))
        else:
            with Image.open(request["rgb"]) as source:
                image = source.convert("RGB")
            output_folder = Path(request["output"])
            output_folder.mkdir(parents=True, exist_ok=True)
        records = []
        with (
            redirect_stdout(sys.stderr),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            loaded = time.perf_counter()
            state = processor.set_image(image)
            encoded = time.perf_counter()
            for index, prompt in enumerate(request["prompts"]):
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(prompt=prompt, state=state)
                if output_folder is not None:
                    masks = (
                        output["masks"]
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1, *image.size[::-1])
                    )
                    np.save(output_folder / f"{index}.npy", masks > 0.5)
                    records.append(
                        {
                            "prompt": prompt,
                            "scores": output["scores"].float().cpu().tolist(),
                            "boxes_xyxy": output["boxes"].float().cpu().tolist(),
                        }
                    )
            # The reply marks completed GPU work, including warmup without CPU output.
            torch.cuda.synchronize()
        finished = time.perf_counter()
        result = {
            "seconds": {
                "load_model": load_seconds,
                "encode_image": encoded - loaded,
                "prompts": finished - encoded,
            },
            "prompts": records,
        }
        if output_folder is not None:
            (output_folder / "instances.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            result["output"] = str(output_folder)
        load_seconds = 0.0
        return result

    if arguments.socket is not None:
        # One request at a time, no polling or background GPU work when idle.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(arguments.socket))
            arguments.socket.chmod(0o600)
            listener.listen()
            print(json.dumps({"ready": True}), flush=True)
            while True:
                connection, _ = listener.accept()
                try:
                    with (
                        connection,
                        connection.makefile("rw", encoding="utf-8") as stream,
                    ):
                        line = stream.readline()
                        if not line:
                            continue
                        try:
                            result = infer(json.loads(line))
                        except Exception as error:  # noqa: BLE001 - report a failed RPC
                            result = {"error": f"{type(error).__name__}: {error}"}
                        stream.write(json.dumps(result) + "\n")
                        stream.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # A cancelled caller must not kill the shared worker.
                    pass
    elif arguments.serve:
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            if line.strip():
                result = infer(json.loads(line))
                print(json.dumps({"output": result["output"]}), flush=True)
    else:
        infer(
            {
                "rgb": str(arguments.rgb),
                "output": str(arguments.output),
                "prompts": arguments.prompts,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
