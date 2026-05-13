#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate opd
which python
python --version
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'devices', torch.cuda.device_count()); print('vllm', vllm.__version__)"
echo "--- /workspace/RESD ---"
ls /workspace/RESD | head
