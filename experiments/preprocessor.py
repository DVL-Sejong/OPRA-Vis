import pandas as pd
import json, csv
import re
from tqdm import tqdm

tqdm.pandas()

class Preprocessor:
    def __init__(self, dataname:str, num_records: int, input_filename: str, output_filename: str, processed_filename: str = None) -> None:
        self.dataname = dataname
        self.num_records = num_records
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.processed_filename = processed_filename

        if dataname == 'amazon':
            self.field = 'sentence'
            self.cleanse_method = self._cleanse_amazon
        elif dataname == 'local':
            self.field = 'text'
            self.cleanse_method =  self._cleanse_plain_text
        elif dataname == 'imdb':
            self.field = 'review'
            self.cleanse_method = self._cleanse_html
        elif dataname == 'jigsaw':
            self.field = 'comment_text'
            self.cleanse_method = self._cleanse_plain_text
        else:
            raise Exception(f'`{dataname}` is unknown data.')

        self.df = self._load_data()
        self.pdf = self._load_processed_data()
    
    def _load_data(self) -> pd.DataFrame:
        # 데이터 로드
        if self.dataname in ['amazon', 'imdb', 'jigsaw']:
            df = pd.read_csv(self.input_filename)
        
        elif self.dataname == 'local':
            with open(self.input_filename, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in tqdm(file, desc='Loading data')]
            textset = [item[self.field] for item in tqdm(data, desc='Fetching data') if self.field in item]
            df = pd.DataFrame(textset, columns=[self.field])
        
        else:
            raise Exception(f'`{self.dataname}` is unknown data.')
        return df
    
    def _load_processed_data(self) -> pd.DataFrame:
        if not self.processed_filename:
            return None

        return pd.read_csv(self.processed_filename)
    
    def preprocess(self) -> None:
        self.df = self.df.dropna()

        print('Cleansing data:')
        self.df[self.field] = self.df[self.field].progress_apply(self.cleanse_method)

        self.df = self.df.drop_duplicates(subset=[self.field])
        self.df = self.df.reset_index(drop=True)

        print('Calculating length:')
        self.df['length'] = self.df[self.field].progress_apply(len)
    
    def get_processed_data(self) -> pd.DataFrame:
        num_records = self.num_records
        
        if self.pdf is not None:
            num_records -= self.pdf.shape[0]
            unprocessed_df = self.df[~self.df[self.field].isin(self.pdf[self.field])]
            print(f"{self.pdf.shape[0]} records loaded from processed data.")
        else:
            unprocessed_df = self.df
        
        if num_records < 0:
            raise Exception("The number of records to return is negative.")

        # 길이 조건 필터링
        unprocessed_df = unprocessed_df[(unprocessed_df['length'] >= 100) & (unprocessed_df['length'] <= 300)]

        # if self.dataname == 'jigsaw':
        #     # 라벨 컬럼 리스트
        #     label_columns = ['obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']

        #     # 0.5를 기준으로 True/False 이진 라벨 생성
        #     for col in label_columns:
        #         unprocessed_df.loc[:, f'{col}_binary'] = unprocessed_df[col] > 0.5
            
        #     # 이진 라벨 컬럼 리스트
        #     binary_label_columns = [f'{col}_binary' for col in label_columns]

        #     # 라벨 조합 컬럼 생성
        #     unprocessed_df['label_combination'] = unprocessed_df[binary_label_columns].apply(tuple, axis=1)

        #     # 각 조합별로 균등하게 샘플링
        #     grouped = unprocessed_df.groupby('label_combination', group_keys=False)
        #     min_samples_per_group = min(len(group) for _, group in grouped) # 가장 작은 그룹 크기 확인
        #     samples_per_group = min(num_records // grouped.ngroups, min_samples_per_group)
        #     sampled_df = grouped.apply(lambda x: x.sample(n=min_samples_per_group, random_state=42) if len(x) >= samples_per_group else x)
        #     # sampled_df = sampled_df.droplevel(0) #.reset_index(drop=True)

        #     # 부족한 데이터 채우기
        #     if len(sampled_df) < num_records:
        #         remaining_df = unprocessed_df[~unprocessed_df.index.isin(sampled_df.index)]
        #         additional_samples = remaining_df.sample(n=num_records - len(sampled_df), random_state=42)
        #         sampled_df = pd.concat([sampled_df, additional_samples])
        # else:
        #     sampled_df = unprocessed_df.sample(frac=1, random_state=42)[:num_records]
        
        # 랜덤 샘플링
        sampled_df = unprocessed_df.sample(frac=1, random_state=42)[:num_records]

        if self.pdf is not None:
            return pd.concat([self.pdf[[self.field]], sampled_df[[self.field]]], ignore_index=True)
        return sampled_df

    def _cleanse_amazon(self, text: str) -> str:
        if pd.isnull(text):
            return text # 결측치는 그대로 반환
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _cleanse_plain_text(self, text: str) -> str:
        if pd.isnull(text):
            return text  # 결측치는 그대로 반환
        text = text.lower()  # 모든 문자를 소문자로 변환
        text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나의 공백으로 변환
        text = re.sub(r'[^\w\s]', '', text)  # 알파벳, 숫자, 공백을 제외한 문자 제거
        return text

    def _cleanse_html(self, text: str) -> str:
        if pd.isnull(text):
            return text  # 결측치는 그대로 반환
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = re.sub(r'\[[^]]*\]', '', text)
        text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나의 공백으로 변환
        return text
