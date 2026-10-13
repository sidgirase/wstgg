cat > run_person2_direct.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

MODEL="smolvlm2_2b"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-./runs_direct_person2_${RUN_TAG}}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available in this session"; print("CUDA:", torch.cuda.get_device_name(0))'

mkdir -p "$RUN_ROOT"

for task in infovqa tabmwp ai2d; do
  for placement in lm vision; do
    for seed in 42 43 44; do
      run_id="direct_${RUN_TAG}_${MODEL}_${task}_${placement}_s${seed}"
      output_dir="${RUN_ROOT}/${run_id}"

      echo "Starting ${run_id}"
      python scripts/run_train.py train \
        --model "$MODEL" \
        --task "$task" \
        --placement "$placement" \
        --seed "$seed" \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --lr 2e-4 \
        --max_steps 500 \
        --mixed_precision bf16 \
        --run_id "$run_id" \
        --output_dir "$output_dir" \
        --manifest_path "${RUN_ROOT}/manifest.json" \
        2>&1 | tee "${RUN_ROOT}/${run_id}.log"
    done
  done
done

echo "Person 2 runs finished. Outputs: ${RUN_ROOT}"
BASH

chmod +x run_person2_direct.sh
./run_person2_direct.sh