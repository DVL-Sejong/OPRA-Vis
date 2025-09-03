import os, sys
import argparse
import dotenv
import pandas as pd
import time
from tqdm import tqdm

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src import *
from experiments.progress import load_progress, save_progress

def confirm_overwrite(file_path):
    """Prompt the user to confirm overwriting an existing file."""
    if os.path.exists(file_path):
        response = input(f'The file {file_path} already exists. Overwrite? (y/n): ')
        if response.lower() != 'y':
            print('Operation cancelled by the user.')
            exit()

dotenv.load_dotenv()

parser = argparse.ArgumentParser(description='Assess OPRA concepts in dataset using LLM.')
parser.add_argument('--model', type=str, help='The model to use for assessment.', choices=('gemma', 'gpt-3', 'llama-2-7b', 'llama-2-13b'))
parser.add_argument('--account', type=str, help='The account number', required=False)
parser.add_argument('--method', type=str, help='The method to use for processing the data.', choices=('vanilla_os', 'vanilla_fs_8', 'cot_os', 'cot_fs_8', 'carp_os', 'carp_fs_8'))
parser.add_argument('--field', type=str, help='Field name from the dataset to process.')
parser.add_argument('--datafile', type=str, help='Path to the data file')
parser.add_argument('--cachefile', type=str, help='Path to the cache file for storing results')

args = parser.parse_args()
model = args.model
account = args.account
method = args.method
field = args.field
datafile = f'experiments/data/{args.datafile}'
cachefile = f'experiments/cache/{args.cachefile}'
# datafile = 'experiments/data/emails_filtered.csv'
# cachefile = 'experiments/cache/carp_email.pkl'

# confirm_overwrite(cachefile)

contents = pd.read_csv(datafile)[field].tolist()
concepts = ['Trust', 'Commitment', 'Control Mutuality', 'Satisfaction']

model_name = 'google/gemma-7b'
account_postfix = f'_{account}' if account else ''
token = os.environ.get(f'GEMMA_TOKEN{account_postfix}', '')
if model == 'gpt-3':
    model_name = 'gpt-3.5-turbo'
    token = os.environ.get(f'OPENAI_API_KEY{account_postfix}', '')
elif model == 'llama-2-7b':
    model_name = 'meta-llama/Llama-2-7b-hf'
    token = os.environ.get(f'LLAMA_TOKEN', '')
elif model == 'llama-2-13b':
    model_name = 'meta-llama/Llama-2-13b-hf'
    token = os.environ.get(f'LLAMA_TOKEN', '')

llm = LLM(model_name, token=token, use_cache=False)

if method not in llm.prompts:
    print('Method does not exist.')
    exit()

progress = load_progress(cachefile)

print('\nAssessment details')
print(f'- Model: {model_name}')
print(f'- Method: {method}')
print(f'- Datafile: {datafile}')
print(f'- Cachefile: {cachefile}\n')

for i in tqdm(range(len(contents)), desc=f'Processing assessment ({model_name})'):
    for concept in tqdm(concepts, desc='Processing concepts', leave=False):
        # 진행 상태 확인하여 이미 처리된 항목은 건너뛰기
        if (i, concept) not in progress:
            llm_data = llm.get_llm_data(concept, contents, [i], method=method, prompt=None, output_attentions=False)
            save_progress(cachefile, i, concept, llm_data)

            # GPT 모델인 경우 10초 쉼
            if model == 'gpt-3':
                time.sleep(10)

print(f'\n\nDone.\nThe data assessed by the LLM has been saved to `{cachefile}`.\n\n')
