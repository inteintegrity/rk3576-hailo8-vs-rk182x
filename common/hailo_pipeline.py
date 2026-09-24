"""HailoRT pipeline for Hailo-8, the way Hailo's own examples drive the device.

Calling one inference and waiting for its result leaves the device idle during everything
that is not the inference call, which on this board costs more than half of the device's
throughput. This module therefore keeps `depth` inferences in flight at all times - the
structure Hailo's documentation prescribes for throughput and what `hailortcli run` does
internally:

    submit(frame)          -> fills the next free slot and starts it, never waits on the device
    collect()              -> waits for the oldest slot and returns its outputs, in order

Slots are recycled in submission order, so a slot's buffers are only handed out again after
its outputs have been collected; outputs are copied out of the slot buffer before reuse.
`hailo_platform` is imported inside the class, so `--help` and static checks work on a host
without HailoRT installed.
"""

from __future__ import annotations

import gc
import os
from collections import deque
from pathlib import Path

import numpy as np

#: HEF head spatial size -> stride, and the channel count that identifies each branch.
STRIDES = {80: 8, 40: 16, 20: 32}
BOX_CHANNELS = 4
SCORE_CHANNELS = 80


class HailoPipeline:
    """Depth-bounded pipelined inference on one Hailo device."""

    def __init__(self, hef_path: Path, depth: int = 4, batch_size: int = 1, vdevice=None):
        from hailo_platform import FormatType, HEF, VDevice

        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.debug = bool(os.environ.get("HAILO_PIPELINE_DEBUG"))
        self._log(f"loading HEF {hef_path}")
        self.hef = HEF(str(hef_path))
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.input_shape = tuple(self.hef.get_input_vstream_infos()[0].shape)
        self.output_names = [info.name for info in self.hef.get_output_vstream_infos()]
        self._owns_device = vdevice is None
        self._log("creating device")
        self._vdevice = vdevice or VDevice()
        self._log("creating infer model")
        self.model = self._vdevice.create_infer_model(str(hef_path))
        self.model.input().set_format_type(FormatType.UINT8)
        for output in self.model.outputs:
            output.set_format_type(FormatType.FLOAT32)
        if batch_size != 1:
            self.model.set_batch_size(batch_size)
        self.batch_size = batch_size
        self._log("configuring")
        self.configured = self.model.configure()
        self._log("activating")
        self.configured.activate()
        self._log("activated")
        self._slots = []
        # HailoRT validates nothing here: a buffer whose shape does not match the stream
        # exactly segfaults the process, so the shapes reported by the model are used
        # verbatim (they carry no leading batch dimension at batch size 1).
        def shape_for(shape) -> tuple:
            shape = tuple(shape)
            return shape if batch_size == 1 else (batch_size, *shape)

        for slot in range(depth):
            self._log(f"binding slot {slot}")
            input_buffer = np.empty(shape_for(self.input_shape), dtype=np.uint8)
            output_buffers = {
                name: np.empty(shape_for(self.model.output(name).shape), dtype=np.float32)
                for name in self.output_names
            }
            bindings = self.configured.create_bindings(
                input_buffers={self.input_name: input_buffer}, output_buffers=output_buffers
            )
            self._slots.append(
                {"bindings": bindings, "input": input_buffer, "outputs": output_buffers}
            )
        self._in_flight: deque = deque()
        self._free = deque(range(depth))

    def submit(self, network_input: np.ndarray, meta=None) -> None:
        """Copy one (or a batch of) NHWC uint8 frame(s) into a free slot and start it."""
        if not self._free:
            raise RuntimeError("no free slot: collect() before submitting beyond the pipeline depth")
        slot_index = self._free.popleft()
        slot = self._slots[slot_index]
        slot["input"][...] = (
            network_input if self.batch_size > 1 else np.squeeze(network_input, axis=0)
        )
        self._log(f"submit slot {slot_index}")
        job = self.configured.run_async([slot["bindings"]])
        self._in_flight.append((slot_index, meta, job))

    def collect(self, timeout_ms: int = 10000):
        """Wait for the oldest submission; return (meta, {output name: copied array})."""
        if not self._in_flight:
            raise RuntimeError("nothing in flight")
        slot_index, meta, job = self._in_flight.popleft()
        self._log(f"wait slot {slot_index}")
        job.wait(timeout_ms)
        self._log(f"waited slot {slot_index}")
        slot = self._slots[slot_index]
        outputs = {name: buffer.copy() for name, buffer in slot["outputs"].items()}
        self._free.append(slot_index)
        return meta, outputs

    def free_slots(self) -> int:
        return len(self._free)

    def in_flight_count(self) -> int:
        return len(self._in_flight)

    def drain(self) -> None:
        """Collect every in-flight inference so nothing is left queued. No teardown.

        Interpreter shutdown cannot be used as a fallback: HailoRT's exit-time destructors
        abort the process. Callers that skip close() must therefore end with os._exit()."""
        while self._in_flight:
            self.collect(timeout_ms=10000)

    def close(self) -> None:
        """Drain, deactivate and release the configured model.

        HailoRT aborts the process if a configured model is still active at exit, so this
        must run before the interpreter shuts down. Note: deactivate()/shutdown() can block
        in this binding while the async callback thread is alive (observed intermittently),
        and they block while holding the GIL - a caller that does not need an orderly
        teardown should use drain() plus os._exit() instead.
        """
        self._log("close: draining")
        self.drain()
        self._log("close: deactivate")
        self.configured.deactivate()
        self._log("close: shutdown")
        self.configured.shutdown()
        # Drop every Python handle to the configured model before releasing the device:
        # this binding segfaults at interpreter shutdown if the model objects outlive the
        # release.
        self._slots = []
        self._free.clear()
        self.configured = None
        self.model = None
        gc.collect()
        if self._owns_device:
            self._vdevice.release()
        self._vdevice = None
        self._log("close: done")

    def _log(self, message: str) -> None:
        if self.debug:
            print(f"[hailo] {message}", flush=True)


def collect_heads(raw_outputs: dict):
    """Identify branch and stride from tensor shape, not layer name.

    YOLO26 heads are 4-channel direct boxes plus 80-channel score logits. HEF layer names
    are compiler-generated (conv61..conv94 here), shapes are what identifies them.
    """
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    for name, array in raw_outputs.items():
        squeezed = np.squeeze(np.asarray(array))
        if squeezed.ndim != 3:
            raise RuntimeError(f"unexpected HEF output rank for {name}: {np.asarray(array).shape}")
        if squeezed.shape[-1] in (BOX_CHANNELS, SCORE_CHANNELS):
            height, _, channels = squeezed.shape
            head = squeezed
        elif squeezed.shape[0] in (BOX_CHANNELS, SCORE_CHANNELS):
            channels, height, _ = squeezed.shape
            head = np.transpose(squeezed, (1, 2, 0))
        else:
            raise RuntimeError(f"cannot identify layout of {name}: {np.asarray(array).shape}")
        if height not in STRIDES:
            raise RuntimeError(f"unexpected spatial size {height} for {name}")
        branch = "box" if channels == BOX_CHANNELS else "score"
        by_stride.setdefault(STRIDES[height], {})[branch] = head

    box_heads, score_heads = [], []
    for stride in sorted(by_stride):
        if set(by_stride[stride]) != {"box", "score"}:
            raise RuntimeError(f"stride {stride}: incomplete head pair {sorted(by_stride[stride])}")
        box_heads.append(by_stride[stride]["box"])
        score_heads.append(by_stride[stride]["score"])
    if len(box_heads) != 3:
        raise RuntimeError(f"expected 3 scales, found {sorted(by_stride)}")
    return box_heads, score_heads