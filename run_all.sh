#!/usr/bin/env bash
# Everything, in priority order: main greedy matrix, temperature slice, noise replicate.
cd "$(dirname "$0")"
./sweep.sh
TEMP=0.7 LENS=128 CONCS=1,16 ./sweep.sh baseline eagle3:4 draft_model:4
LABEL_SUFFIX=_rep LENS=128 CONCS=1,16 PROFILE_CONFIGS=none ./sweep.sh baseline eagle3:4
echo ALL_DONE
