# Kaggle v10

Use `v10_config.json` from the Kaggle notebook with:

```bash
uv run python train.py --config-json kaggle/v10_config.json --hf
```

Before launching the long run:

- Stop any active Kaggle session.
- Select GPU accelerator `T4 x2`.
- Ensure Kaggle secret `HF_TOKEN` is available.
- Place or download `v10_pretrain.npz` at `/kaggle/working/data/curated/v10_pretrain.npz`.
- Keep the run id as `policy_spatial_v10`.

Do not use `hf_bootstrap_run_id` for the official v10 run. The model starts from
scratch and uses curated pretraining before self-play.
