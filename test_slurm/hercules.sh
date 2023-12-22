#!/bin/bash
for ME in "1000"
do
    for dataset in "CIFAR10"
    do
        for dim in "8" 
        do
            
            for sigma in "2" "4" "6" "8" "10" "12" "14" "16" "18" "20"
            do
                echo   "sbatch -o /dev/null \
                        -J sigma_${sigma} \
                        -c 1 \
                        #-x node[01-12] \
                        ./test_slurm/general_script.sh $dataset $dim $ME LIME $sigma &"

                sbatch -o /dev/null \
                        -J sigma_${sigma} \
                        -c 1 \
                        ./test_slurm/general_script.sh $dataset $dim $ME LIME $sigma &
                
            done
        done
    done
done
# Imprime el número de jobs que se están ejecutando
echo "Number of jobs: $(squeue -u $USER | wc -l)"

