import os, sys
import argparse
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from experiments.postprocessor import Postprocessor

def confirm_overwrite(file_path):
    """Prompt the user to confirm overwriting an existing file."""
    if os.path.exists(file_path):
        response = input(f'The file {file_path} already exists. Overwrite? (y/n): ')
        if response.lower() != 'y':
            print('Operation cancelled by the user.')
            exit()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, choices=('gemma', 'gpt-3', 'llama-2'))
parser.add_argument('--method', type=str, choices=('vanilla_os', 'vanilla_fs_8', 'cot_os', 'cot_fs_8', 'carp_os', 'carp_fs_8'))
parser.add_argument('--dataname', type=str, choices=('amazon', 'local', 'imdb'))
parser.add_argument('--progress', type=str, required=False, default=None)
parser.add_argument('--input', type=str, required=False, default=None)
parser.add_argument('--output', type=str, required=False, default=None)

args = parser.parse_args()
model_prefix: str = ''
if args.model == 'gpt-3':
    model_prefix = 'gpt-3_'
elif args.model == 'llama-2':
    model_prefix = 'llama-2-7b_'
progress_filename = f'experiments/cache/{model_prefix}{args.method}_{args.dataname}.pkl' if not args.progress else args.progress
input_filename = f'experiments/data/pre/{args.dataname}_filtered.csv' if not args.input else args.input
output_filename = f'experiments/data/llm/{model_prefix}{args.method}_{args.dataname}.csv' if not args.output else args.output

confirm_overwrite(output_filename)

postprocessor = Postprocessor(
    method=args.method,
    dataname=args.dataname,
    progress_filename=progress_filename,
    input_filename=input_filename
)
postprocessor.postprocess()
df = postprocessor.get_processed_data()
df.to_csv(output_filename)

print(f'The postprocessed data is stored in `{output_filename}`.')
