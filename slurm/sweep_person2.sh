#!/bin/bash
# Submits all runs assigned to Person 2 (SmolVLM2-2.2B across InfographicVQA, TabMWP, AI2D, 3 seeds)
bash slurm/sweep.sh "Person_2" "smolvlm2_2b"
