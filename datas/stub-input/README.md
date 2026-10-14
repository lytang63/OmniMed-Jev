# stub-input

`train`, `infer` and `evaluate` read their rows from
`{output_dir}/Preprocessed-{dataset}/`, not from here. What they *do* need from
`--input_dir` is a directory that exists and looks like a dataset, because the
sanity check in `omnimedjev/engine/check_dataset.py` refuses to run otherwise.

This is that directory, and nothing else. It carries no data: the single stub
row is never read. Point `--input_dir` here when the training rows live in
`--output_dir` (which is what the generative arm does), and point it at the real
converted tree during `preprocess`.
