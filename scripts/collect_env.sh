#!/usr/bin/env bash
# Record the software/hardware environment into results/env.txt
source "${VENV:-/root/venv}/bin/activate"
cd "$(dirname "$0")/.."
{
  echo "date: $(date -u)"
  python -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__)"
  nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv
  nvidia-smi topo -m | head -5
  lscpu | grep -E "Model name|^CPU\(s\)"
  free -g | head -2
} > results/env.txt 2>&1
cat results/env.txt
