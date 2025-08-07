## reproduce DABStep baselines

### run with azure identity token:
    * for baseline: `python baseline/run.py --model-id "o3-mini" --use-azure-auth --max-tasks 2 --split default --concurrency 1`
    * for 2 cluster examples: `python baseline/run2.py --model-id "o3-mini" --use-azure-auth --max-tasks 2 --split default --concurrency 1`

### 