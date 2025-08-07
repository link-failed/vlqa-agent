## reproduce DABStep baselines

### env configuration
    * enter conda env
    * `install -r baseline/requirements.txt`
    * `az login` (if use azure identity token)

### run with azure identity token (batch test):    
    * for baseline: `python baseline/run.py --model-id "o3-mini" --use-azure-auth --max-tasks 2 --split default --concurrency 1`
    * for 2 cluster examples: `python baseline/run2.py --model-id "o3-mini" --use-azure-auth --max-tasks 2 --split default --concurrency 1`

### run with azure identity token (single test):
    * `python baseline/run.py --model-id $MODEL_ID --use-azure-auth --tasks-ids 49 5 1273 --split default --concurrency 3`