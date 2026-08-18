#!/usr/bin/env python3
from argparse import ArgumentParser
from safesense.data import prepare_utd_mhad

p = ArgumentParser(description="Prepare UTD-MHAD skeleton+IMU arrays for SafeSense")
p.add_argument("--data-dir", required=True, help="Directory containing extracted Skeleton/ and Inertial/ MAT files")
p.add_argument("--output-dir", default="data/processed")
a = p.parse_args()
print(prepare_utd_mhad(a.data_dir, a.output_dir))
