#!/bin/bash
# Master script to submit all training jobs with dependencies
# This ensures jobs run in the correct order

# Submit the pre-pretraining job on shuff_dyck
job1=$(sbatch --parsable scripts/slurm_pretrain_shuff_dyck.sh)
echo "Submitted pre-pretraining on shuff_dyck: Job ID $job1"

# Submit the vanilla c4 pretraining job (independent, can run in parallel)
job2=$(sbatch --parsable scripts/slurm_pretrain_c4.sh)
echo "Submitted vanilla pretraining on c4: Job ID $job2"

# Submit the c4 pretraining with shuff_dyck initialization (depends on job1)
job3=$(sbatch --parsable --dependency=afterok:$job1 scripts/slurm_pretrain_c4_shuff_dyck.sh)
echo "Submitted c4 pretraining with shuff_dyck initialization: Job ID $job3 (depends on $job1)"

echo ""
echo "All jobs submitted!"
echo "Job 1 (pre-pretrain shuff_dyck): $job1"
echo "Job 2 (vanilla c4): $job2"
echo "Job 3 (c4 w/ shuff_dyck): $job3 (waits for Job 1)"
