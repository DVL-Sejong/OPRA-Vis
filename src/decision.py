import torch
import os
import pickle
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class LLM:
    def __init__(self, model_name: str, token: str=None, use_cache: bool=True, no_gpu: bool=False, **kwargs) -> None:
        """
        LLM 클래스 초기화

        Args:
            model_name (str): 사용할 모델 이름
        """

        self.model_name = model_name
        self.use_cache = use_cache
        self.no_gpu = no_gpu
        if not self.no_gpu:
            self.generator = TextGenerator(model_name, token=token, **kwargs)
        self.prompts = {
            'carp_os': self._load_prompts("src/prompts/carp_os"),
            'carp_fs_8': self._load_prompts("src/prompts/carp_fs_8"),
            'cot_os': self._load_prompts("src/prompts/cot_os"),
            'cot_fs_8': self._load_prompts("src/prompts/cot_fs_8"),
            'vanilla_os': self._load_prompts("src/prompts/vanilla_os"),
            'vanilla_fs_8': self._load_prompts("src/prompts/vanilla_fs_8"),
        }

        if use_cache:
            self.llm_cache = self._load_llm_cache("src/cache/llm_cache.pkl")

    def _load_prompts(self, directory: str) -> dict[str, str]:
        """
        사전 정의된 프롬프트 로드

        Args:
            directory (str): 프롬프트 파일이 저장된 디렉토리 경로

        Returns:
            dict[str, str]: 파일 이름(확장자 제외)을 키로 하고 파일 내용을 값으로 하는 딕셔너리
        """

        prompts = {}
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)

                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()

                name_without_extension = os.path.splitext(filename)[0]
                prompts[name_without_extension] = content
        return prompts
    
    def _load_llm_cache(self, filename: str) -> dict:
        """
        캐시 파일에서 이전에 저장된 LLM 데이터를 로드

        Args:
            filename (str): 캐시 파일 이름

        Returns:
            dict: 캐시된 LLM 데이터
        """

        self.llm_cache_filename = filename
        # 캐시 파일 존재 시 로드
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                return pickle.load(file)
        return {}
    
    def _save_llm_cache(self) -> None:
        """
        현재 LLM 캐시를 파일에 저장
        """

        with open(self.llm_cache_filename, 'wb') as file:
            pickle.dump(self.llm_cache, file)

    def get_llm_data(self, concept: str, contents: list[str], content_id_list: list[int], method: str, prompt: str | None, output_attentions: bool = True) -> list[dict]:
        """
        주어진 concept에 대해 콘텐츠를 LLM으로 평가를 생성하거나 캐시에서 불러옴

        Args:
            concept (str): 분석할 concept
            contents (list[str]): 분석할 콘텐츠의 리스트
            content_id_list (list[int]): 분석할 콘텐츠의 ID 목록
            method (str): 프롬프팅 방법
            prompt (str | None): 사용할 프롬프트
            output_attentions (bool): Attention 정보 반환 여부

        Returns:
            list[dict]: 각 콘텐츠에 대한 LLM의 평가 결과
        """

        llm_data = []
        for content_id in content_id_list:
            if prompt is None: # 프롬프트 preset 사용할 경우
                used_prompt = self.prompts[method][concept]
            else:
                used_prompt = prompt
            
            # 프롬프트 preset 사용하기 위해 formatting
            input_text = used_prompt.format(content=contents[content_id].strip())

            # 캐싱된 결과에 대해서는 다시 생성하지 않음
            cache_key = (self.model_name, concept, contents[content_id], input_text) # 캐시 키
            if self.use_cache and cache_key in self.llm_cache:
                llm_data.append(self.llm_cache[cache_key])
                continue
            if self.no_gpu:
                continue

            if output_attentions:
                # LLM으로 텍스트 생성
                text, attentions, tokenizer, input_ids, outputs = self.generator.generate_text(
                    input_text=input_text,
                    max_new_tokens=200,
                    stop_newline=False,
                )

                # Inter-Sentence Attention 연산
                isa = InterSentenceAttention(input_ids, outputs.sequences[0], attentions, tokenizer)
                result = {
                    'content_id': content_id,
                    'text': text,
                    'prompt_text': used_prompt,
                    'generated_text': text[len(input_text):],
                    'num_prompt_sentences': isa.num_prompt_sentences,
                    'num_genereated_sentences': isa.num_generated_sentences,
                    'sentences': [{
                        'sentence': sentence,
                        'isa': scores.tolist(),
                    } for sentence, scores in zip(isa.sentences, isa.sentence_attention)],
                }

                if self.use_cache:
                    self.llm_cache[cache_key] = result # 결과를 캐싱
                    self._save_llm_cache() # 캐시 파일 저장

                llm_data.append(result)
            else:
                text, generated_text = self.generator.generate_text(
                    input_text=input_text,
                    max_new_tokens=200,
                    stop_newline=False,
                    output_attentions=False,
                )

                result = {
                    'content_id': content_id,
                    'text': text,
                    'prompt_text': used_prompt,
                    'generated_text': generated_text,
                }

                llm_data.append(result)

        return llm_data

class TextGenerator:
    def __init__(self, model_name: str, token: str=None, **kwargs):
        if model_name.startswith('gpt-'):
            self.generator = OpenAITextGenerator(model_name, token, kwargs=kwargs)
        else:
            self.generator = HuggingFaceTextGenerator(model_name, token, **kwargs)
    
    def generate_text(self, input_text: str, max_new_tokens: int=200, stop_newline: bool=False, output_attentions: bool=True):
        return self.generator.generate_text(
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            stop_newline=stop_newline,
            output_attentions=output_attentions,
        )

class HuggingFaceTextGenerator:
    def __init__(self, model_name: str, token: str=None, **kwargs):
        token = token or os.environ.get("GEMMA_TOKEN", "")
        if not token:
            raise ValueError("A token for Gemma is required. Please specify a token.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", output_attentions=True, token=token, **kwargs)

        print('`HuggingFaceTextGenerator` class loaded.')

    def generate_text(self, input_text: str, max_new_tokens: int=200, stop_newline: bool=False, output_attentions: bool=True):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        if stop_newline:
            stopping_criteria = StoppingCriteriaList([StopOnNewLineCriteria(self.tokenizer)])
        else:
            stopping_criteria = None
        outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, output_attentions=True, return_dict_in_generate=True, stopping_criteria=stopping_criteria)
        text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        attentions = outputs.attentions

        if output_attentions:
            return text, attentions, self.tokenizer, input_ids, outputs
        return text, text[len(input_text):]

class OpenAITextGenerator:
    def __init__(self, model_name: str, token: str=None, **kwargs):
        self.model_name = model_name

        openai.api_key = token or os.environ.get("OPENAI_API_KEY", "")
        if not openai.api_key:
            raise ValueError("An API key for OpenAI is required. Please specify an API key.")
        
        print('`OpenAITextGenerator` class loaded.')
    
    def generate_text(self, input_text: str, max_new_tokens: int=200, stop_newline: bool=False, output_attentions: bool=True):
        outputs = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'user', 'content': input_text}
            ],
            max_tokens=max_new_tokens,
            stop=['\n'] if stop_newline else None,
        )
        generated_text = str(outputs.choices[0].message.content).strip()

        text = input_text + generated_text

        if output_attentions:
            return text, None, None, None, outputs
        return text, generated_text

class StopOnNewLineCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # 마지막으로 생성된 토큰이 줄바꿈인지 확인
        if input_ids[0, -1] == self.tokenizer.eos_token_id or \
            self.tokenizer.decode(input_ids[0, -1]) == '\n':
            return True
        return False

class InterSentenceAttention:
    def __init__(self, input_ids, generated_sequences, attentions, tokenizer):
        self.input_ids = input_ids
        self.tokenizer = tokenizer

        # 문장을 마침표 기준으로 분리
        sentence_boundaries, sentences, num_prompt_sentences, num_generated_sentences = self._find_sentence_boundaries(generated_sequences)

        # 문장 간 attention 계산
        integrated_attentions = self._integrate_attentions(attentions)
        sentence_attention_heads = self._calculate_sentence_attention(integrated_attentions, sentence_boundaries)
        sentence_attention = self._aggregate_attention(sentence_attention_heads)

        self.sentences = sentences
        self.sentence_attention_heads = sentence_attention_heads
        self.sentence_attention = sentence_attention
        self.num_prompt_sentences = num_prompt_sentences
        self.num_generated_sentences = num_generated_sentences
    
    def _integrate_attentions(self, attentions):
        num_layers = len(attentions)
        _, num_heads, prompt_seq_length, _ = attentions[0][-1].shape
        max_seq_length = attentions[-1][-1].shape[-1]

        integrated = torch.zeros((1, num_heads, max_seq_length, max_seq_length))
        integrated[0, :, :prompt_seq_length, :prompt_seq_length] = attentions[0][-1] # 0번째 레이어(프롬프트 간의 attention을 가짐)의 마지막 히든 레이어의 heads들은 그대로 붙임
        # 나머지 레이어들(생성된 토큰들과 이전 토큰들의 attention을 가짐)의 마지막 히든 레이어의 heads들을 이어붙임
        for layer in range(1, num_layers):
            integrated[0, :, prompt_seq_length+layer:prompt_seq_length+layer+1, :prompt_seq_length+layer] = attentions[layer][-1][0]
        
        return integrated # shape: (1, num_heads, max_seq_length, max_seq_length)
    
    def _aggregate_attention(self, sentence_attentions):
        # input shape: (1, num_heads, num_sentences, num_sentences)
        max_attention_heads, _ = torch.max(sentence_attentions, dim=1)

        return max_attention_heads.squeeze(0) # shape: (num_sentences, num_sentences)

    def _find_sentence_boundaries(self, sequences):
        boundaries = [0]
        sentences = []

        sentence_index = 0
        num_prompt_sentences = 0
        num_generated_sentences = 0

        for i, token in enumerate(sequences):
            decoded = self.tokenizer.decode(token)
            if '\n' in decoded:
                sentence = self.tokenizer.decode(sequences[boundaries[-1]:i])

                if i >= self.input_ids.shape[1] and sentence.strip().startswith('INPUT:'):
                    break # 할루시네이션 생성 시 break

                sentences.append(sentence)
                boundaries.append(i)
                sentence_index += 1
                if i < self.input_ids.shape[1]:
                    num_prompt_sentences = sentence_index
        num_generated_sentences = sentence_index - num_prompt_sentences + 1

        # sentences.append(self.tokenizer.decode(sequences[boundaries[-1]:len(sequences)]))
        # boundaries.append(len(sequences))

        return boundaries, sentences, num_prompt_sentences, num_generated_sentences

    def _calculate_sentence_attention(self, attentions, sentence_boundaries):
        # 문장 간 attention 계산
        # num_layers = len(attentions)
        # num_layers = 1
        _, num_heads, _, seq_length = attentions.shape # shape: (1, num_heads, seq_length, seq_length)
        num_sentences = len(sentence_boundaries) - 1 # 시작점과 끝점을 포함하므로 1 뺌

        # 0으로 초기화
        sentence_attentions = torch.zeros((1, num_heads, num_sentences, num_sentences))

        # for layer in range(num_layers):
        # layer = 0
        for head in range(num_heads):
            for i in range(num_sentences):
                start_i = sentence_boundaries[i]
                end_i = sentence_boundaries[i + 1]
                for j in range(num_sentences):
                    start_j = sentence_boundaries[j]
                    end_j = sentence_boundaries[j + 1]
                    # 문장 범위 내 텐서 추출
                    attention_slice = attentions[0, head, start_i:end_i, start_j:end_j]
                    
                    if attention_slice.numel() == 0: # 빈 텐서 검사
                        max_attention = 0
                    else:
                        max_attention = torch.max(attention_slice).item() # 문장 내 토큰들의 attention의 최대값
                    
                    sentence_attentions[0, head, i, j] = max_attention # 각 헤드 내에서 i번째 문장관 j번째 문장의 attention

        return sentence_attentions # shape: (1, num_heads, num_sentences, num_sentences)
