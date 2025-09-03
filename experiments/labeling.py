import os, sys
import argparse
import pandas as pd
from tqdm import tqdm

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.constants import CONCEPTS, CONCEPT_COLUMN_RENAME
from src.scatter_plot import ScatterPlot

tqdm.pandas()

def confirm_overwrite(file_path):
    """Prompt the user to confirm overwriting an existing file."""
    if os.path.exists(file_path):
        response = input(f'The file {file_path} already exists. Overwrite? (y/n): ')
        if response.lower() != 'y':
            print('Operation cancelled by the user.')
            exit()

parser = argparse.ArgumentParser()
parser.add_argument('--field', type=str)
parser.add_argument('--scale', type=str)
parser.add_argument('--dataname', type=str)
parser.add_argument('--datafile', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()
field: str = args.field
scale: str = args.scale
dataname: str = args.dataname
datafile: str = f'experiments/data/pre/{args.datafile}'
outputfile: str = f'experiments/data/label/{args.output}'

confirm_overwrite(outputfile)

df = pd.read_csv(datafile, index_col=0)

scatter_plot = ScatterPlot(dataname=dataname, scale=scale, path=datafile, field=field)
data = scatter_plot.get_scatter_plot_data()

for concept in CONCEPTS[scale]:
    print(f'Labeling `{concept}`')
    df[concept] = [item['opra'][concept] for item in data]
    df[concept] = df[concept].progress_apply(lambda x: 1 if x > 0.5 else 0)

df.to_csv(outputfile)
print("\n\nDone.")
