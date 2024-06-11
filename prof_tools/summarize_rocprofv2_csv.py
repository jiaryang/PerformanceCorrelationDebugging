#!/usr/bin/env python3
import argparse
import re
import csv
import pandas as pd

def add_parser_arguments(parser):
    parser.add_argument('-i', '--input-file', required=True, type=str, help='')
    parser.add_argument('-o', '--output-file', type=str, default='results.stats.csv', help='')
    parser.add_argument('--hcc', action="store_true", help="")

parser = argparse.ArgumentParser(description='Summarize rocprofv2 CSV files')
add_parser_arguments(parser)
args = parser.parse_args()

df = pd.read_csv(args.input_file)

if args.hcc:
    df = df[(df['Operation'] == 'KernelExecution')]
    df['duration'] = df['Stop_Timestamp'] - df['Start_Timestamp']
else:
    df['duration'] = df['End_Timestamp'] - df['Start_Timestamp']
stats = df.groupby('Kernel_Name').agg(TotalDurationNs=('duration', 'sum'), Occurrence=('Kernel_Name', lambda x : len(x)))
stats.columns = [''.join(col).strip() for col in stats.columns.values]
stats.to_csv(args.output_file, index=True)
