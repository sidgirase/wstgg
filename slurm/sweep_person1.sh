#!/bin/bash
# Submits all runs assigned to Person 1 (InternVL3-2B across InfographicVQA, TabMWP, AI2D, 3 seeds)
bash slurm/sweep.sh "Person_1" "internvl3_2b"
