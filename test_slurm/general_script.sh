#!/bin/bash
#SBATCH --partition muylarga

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate /home/isevillano/environments/py-38

python -u ./ReVel/test_revel.py \
    --dataset $1 \
    --dim $2 \
    --nexplaination 5 \
    --max_examples $3 \
    --samples 100 \
    --xai_model $4 \
    --sigma $5