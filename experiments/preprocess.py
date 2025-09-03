import os, sys
import argparse

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from experiments.preprocessor import Preprocessor

def confirm_overwrite(file_path):
    """Prompt the user to confirm overwriting an existing file."""
    if os.path.exists(file_path):
        response = input(f'The file {file_path} already exists. Overwrite? (y/n): ')
        if response.lower() != 'y':
            print('Operation cancelled by the user.')
            exit()

parser = argparse.ArgumentParser(description='')
# parser.add_argument('--method', type=str, choices=('vanilla', 'cot', 'carp'))
parser.add_argument('--dataname', type=str, choices=('amazon', 'local', 'imdb', 'jigsaw'))
parser.add_argument('--records', type=int)
parser.add_argument('--processed', type=str, required=False, default=None)
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()
processed_filename = f'experiments/data/pre/{args.processed}' if args.processed else None
input_filename = f'experiments/data/raw/{args.input}'
output_filename = f'experiments/data/pre/{args.output}'
if output_filename == processed_filename:
    output_filename = f'{output_filename}_{args.records}'

confirm_overwrite(output_filename)

preprocessor = Preprocessor(
    dataname=args.dataname,
    num_records=args.records,
    input_filename=input_filename,
    output_filename=output_filename,
    processed_filename=processed_filename
)
preprocessor.preprocess()
df = preprocessor.get_processed_data()
df.to_csv(output_filename)

print(f'The preprocessed data is stored in `{output_filename}`.')
