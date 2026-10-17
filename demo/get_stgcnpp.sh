#!/usr/bin/env bash
# Downloads the pretrained PYSKL ST-GCN++ weights (NTU-120 cross-subject, HRNet 2D skeletons; Apache-2.0).
set -e
cd "$(dirname "$0")"
BASE=https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet
for f in j b; do
  [ -s stgcnpp_$f.pth ] || curl -fL --retry 3 -o stgcnpp_$f.pth "$BASE/$f.pth"
done
ls -l stgcnpp_*.pth
