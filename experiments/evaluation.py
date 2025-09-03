import os, sys
import argparse
import pandas as pd

if os.getcwd()not in sys.path:
    sys.path.append(os.getcwd())

def confirm_overwrite(file_path):
    '''Prompt the user to confirm overwriting an existing file.'''
    if os.path.exists(file_path):
        response = input(f'The file {file_path} already exists. Overwrite? (y/n): ')
        if response.lower() != 'y':
            print('Operation cancelled by the user.')
            exit()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, choices=('gemma', 'gpt-3', 'llama-2'))
parser.add_argument('--dataname', type=str, choices=('amazon', 'local', 'imdb'))

args = parser.parse_args()
dataname: str = args.dataname
# output_filename: str = args.output

if dataname == 'amazon':
    field = 'sentence'
elif dataname == 'local':
    field = 'text'
elif dataname == 'imdb':
    field = 'review'
else:
    raise Exception(f'`{dataname}` is unknown data.')

model_prefix: str = ''
if args.model == 'gpt-3':
    model_prefix = 'gpt-3_'
elif args.model == 'llama-2':
    model_prefix = 'llama-2-7b_'

methods = ('vanilla_os', 'vanilla_fs_8', 'cot_os', 'cot_fs_8', 'carp_os', 'carp_fs_8')
llm_columns = ('llm_trust', 'llm_satisfaction', 'llm_commitment', 'llm_control mutuality')
llm_df = {method: pd.read_csv(f'experiments/data/llm/{model_prefix}{method}_{dataname}.csv') for method in methods}

humans = ['kim', 'lee']
human_columns = ('Trust', 'Satisfaction', 'Commitment', 'Control Mutuality')
human_df = {name: pd.read_csv(f'experiments/data/human/{name}_{dataname}.csv') for name in humans}

for method in methods:
    print(f'Method: {method}')
    for human in humans:
        print(f'  Human: {human}')
        llm_data = llm_df[method]
        human_data = human_df[human]

        llm_data = llm_data.replace(0.5, 0).fillna(1)
        human_data = human_data.replace(0.5, 0).fillna(1)
        # llm_data = llm_data.replace(0.5, 0).fillna(0)
        # human_data = human_data.replace(0.5, 0).fillna(0)

        avg_accuracy = 0

        for llm_col, human_col in zip(llm_columns, human_columns):
            # 예측값과 실제값이 일치하는 경우의 수 계산
            correct_count = (llm_data[llm_col] == human_data[human_col]).sum()

            # 전체 비교 가능한 항목 수
            total_count = len(llm_data)
            
            # 정확도 계산
            accuracy = correct_count / total_count if total_count > 0 else 0
            avg_accuracy += accuracy
            
            # 컬럼별 정확도 출력
            print(f'    {human_col}: {accuracy:.2f}')
        
        avg_accuracy /= len(human_columns)
        print(f'    Avg.: {avg_accuracy:.2f}')
