import os
import pickle
import pandas as pd
from tqdm import tqdm

class Postprocessor:
    def __init__(self, method: str, dataname: str, progress_filename: str, input_filename: str) -> None:
        self.method = method
        self.dataname = dataname
        self.progress_filename = progress_filename
        self.input_filename = input_filename

        if dataname == 'amazon':
            self.field = 'sentence'
        elif dataname == 'local':
            self.field = 'text'
        elif dataname == 'imdb':
            self.field = 'review'
        elif dataname == 'jigsaw':
            self.field = 'comment_text'
        else:
            raise Exception(f'`{dataname}` is unknown data.')
        
        self.progress: dict[tuple[int, str], dict] = self._load_progress()
        self.idf: pd.DataFrame = self._load_input_data()
        
    def _load_progress(self) -> dict:
        if os.path.exists(self.progress_filename):
            with open(self.progress_filename, 'rb') as f:
                return pickle.load(f)
        raise Exception(f'Progressed cache of the data `{self.dataname}` is not found.')

    def _load_input_data(self) -> pd.DataFrame:
        if not self.input_filename:
            raise Exception(f'`{self.input_filename}` is not found.')
        return pd.read_csv(self.input_filename)
    
    def _parse_encoding(self, text: str) -> int:
        # 텍스트의 마지막 행
        lines = text.splitlines()
        dtext = lines[-1] if lines else text

        # 검색어 및 해당 감정 값 매핑
        search_terms = {
            ('True', '1'): 1,
            ('Neutral', 'Mixed', 'True or False', '0.5'): 0.5,
            ('False', '0'): 0
        }

        # 각 검색어에 대해 마지막 인덱스 찾기 및 해당 감정 값과 매핑
        last_index_and_value = [
            (max(dtext.rfind(term) for term in terms), value)
            for terms, value in search_terms.items()
        ]

        # 가장 마지막 인덱스와 그에 해당하는 감정 값 찾기
        last_emotion_index, emotion_value = max(last_index_and_value, key=lambda x: x[0])

        # 모든 검색어가 없는 경우, None 반환
        if last_emotion_index == -1:
            return None

        return emotion_value

    def postprocess(self) -> list[dict]:
        data_dict = {}

        for (id, concept), progress_data in tqdm(self.progress.items(), desc='Processing data'):
            for item in progress_data:
                content: str = self.idf.loc[id, self.field]
                content_id: int = int(item['content_id'])
                prompt: str = item['prompt_text'].strip()
                generated_text: str = item['generated_text'].split('INPUT: ', 1)[0].strip()

                encoding = self._parse_encoding(generated_text)

                if content_id not in data_dict:
                    data_dict[content_id] = {
                        'content_id': content_id,
                        'content': content,
                    }
                data_dict[content_id][f'llm_{concept.lower()}'] = encoding
                data_dict[content_id][f'llm_{concept.lower()}_reason'] = generated_text
                data_dict[content_id][f'llm_{concept.lower()}_prompt'] = prompt.format(content=content)

        self.data = list(data_dict.values())

    def get_processed_data(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        df = df.set_index('content_id')
        return df
