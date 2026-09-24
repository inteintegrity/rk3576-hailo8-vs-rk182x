# Results

The main branch keeps only the compact benchmark summary and the two final charts needed by the
project page:

```text
results/
  summary.json
  figures/
    single_stream_bars.png
    multi_stream_scaling.png
```

Full per-frame detection records, raw multi-stream JSON, screenshots, articles and the scripts used
to regenerate and verify the figures are attached to the
[`v1.0.0` GitHub Release](https://github.com/inteintegrity/rk3576-hailo8-vs-rk182x/releases/tag/v1.0.0)
as `rk3576-hailo8-vs-rk182x-benchmark-artifacts-v1.0.0.zip`.

Release asset SHA256:

```text
336b595e40355e064ff7f6967000e8bf6adeba5b3339fd6f5f2e571a11220dc8
```

All three single-stream records used the same 640×640, 394-frame sample clip shipped as
`video/test.mp4` (SHA256 `30ec4406e62d37a164c073d44f71d6a7cad70a8eddd26daf5a792a0eba2a2218`).
