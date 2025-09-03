import os, sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import nltk
from itertools import product
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from multiprocessing import Process
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.constants import CONCEPTS, DATA_TEXT_FIELD, CONCEPT_COLUMN_RENAME, CLUE_LABELS, SENTIMENT_LABELS

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=str)
parser.add_argument('--volume', type=int)
parser.add_argument('--dataname', type=str)

args = parser.parse_args()
scale: str = args.scale
volume: int = int(args.volume)
dataname: str = args.dataname
inputfile: str = f'experiments/data/label/{dataname}_{volume}_labeled.csv' # 레이블링된 데이터
outputpath: str = f'experiments/data/sentiment/{dataname}_{volume}'

# outputpath 디렉토리가 없으면 생성
if not os.path.exists(outputpath):
    os.makedirs(outputpath)

# GPU 리스트
num_gpus = torch.cuda.device_count()
gpu_list = [i for i in range(num_gpus)]

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(filtered_words)

def classify_emotion(text, model, tokenizer, device):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        prediction = model(**tokens)
    
    prediction = F.softmax(prediction.logits, dim=1)
    output = prediction.argmax(dim=1).item()
    return output

def process_concept(concept_group, gpu_id, scale, df, outputpath, tokenizer):
    # GPU 설정
    desc_prefix = f'GPU {gpu_id}'
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model_name = 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(SENTIMENT_LABELS)).to(device)

    for clue_id, concept in tqdm(concept_group, desc=f'{desc_prefix} Processing Concepts', leave=True, position=gpu_id*2):
        sentences = df[df[concept] == clue_id]['clean_doc'].values

        word_list = []
        for sentence in tqdm(sentences, desc=f'{desc_prefix} Collecting Words', leave=False, position=gpu_id*2+1):
        # for sentence in sentences:
            words = sentence.split()
            word_list.extend(words)
        word_set = list(set(word_list))

        words = {}
        for word1 in tqdm(word_set, desc=f'{desc_prefix} Counting Words', leave=False, position=gpu_id*2+1):
        # for word1 in word_set:
            cnt = sum(word1 == word2 for word2 in word_list)
            words[word1] = cnt
        
        words = sorted(words.items(), key=lambda item: item[1], reverse=True)
        sentiment_words = [[] for _ in range(len(SENTIMENT_LABELS))]

        for word in tqdm(words, desc=f'{desc_prefix} Classifying Sentiments', leave=False, position=gpu_id*2+1):
        # for word in words:
            sentiment = classify_emotion(word[0], model, tokenizer, device)
            sentiment_words[sentiment].append(word)
        
        for sentiment_id in tqdm(range(len(SENTIMENT_LABELS)), desc=f'{desc_prefix} Saving Results', leave=False, position=gpu_id*2+1):
        # for sentiment_id in range(len(SENTIMENT_LABELS)):
            sentiment_df = pd.DataFrame(sentiment_words[sentiment_id], columns=['sentence', 'frequency'])
            concept_reverse = CONCEPT_COLUMN_RENAME.get_reverse(scale, concept)
            sentiment_df.to_csv(f'{outputpath}/{concept_reverse}_{CLUE_LABELS[clue_id]}_{SENTIMENT_LABELS[sentiment_id]}.csv', index=False)

def split_work(concepts, num_gpus):
    # Concepts를 GPU 수에 따라 분할
    concept_groups = [[] for _ in range(num_gpus)]
    for i, concept in enumerate(concepts):
        concept_groups[i % num_gpus].append(concept)
    return concept_groups

# 데이터 로드
df = pd.read_csv(inputfile, index_col=0)
df['clean_doc'] = df[DATA_TEXT_FIELD[dataname]].str.replace('[^a-zA-Z]', ' ', regex=True)
df['clean_doc'] = df['clean_doc'].apply(lambda x: ' '.join([w.lower() for w in x.split() if len(w) > 3]))

# 컨셉 그룹 분할
concepts = list(product(range(len(CLUE_LABELS)), CONCEPTS[scale]))
concept_groups = split_work(concepts, len(gpu_list))

# 프로세스 생성 및 실행
processes = []
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for gpu_id, concept_group in enumerate(concept_groups):
    print(f'Processing on GPU {gpu_id}: {concept_group}')
for gpu_id, concept_group in enumerate(concept_groups):
    p = Process(target=process_concept, args=(concept_group, gpu_list[gpu_id], scale, df, outputpath, tokenizer))
    processes.append(p)
    p.start()

# 모든 프로세스 완료 대기
for p in processes:
    p.join()

print('Processing completed!')
