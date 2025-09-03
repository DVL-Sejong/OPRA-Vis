import os
import pandas as pd
import numpy as np
import torch
import pickle
import scipy
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from bert_score import score, BERTScorer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.constants import USE_CACHE, CONCEPTS, CONCEPT_COLUMN_RENAME, CLUE_LABELS

class ScatterPlot:
    def __init__(
        self,
        dataname: str,
        scale: str,
        path: str,
        llm_labeled_path: str=None,
        ground_truth_path: str=None,
        field: str='sentence',
        scoring_method: str='coc',
        scaling_method: str='standard',
        adjustment_method: str='gravity',
    ) -> None:
        """
        ScatterPlot 객체 초기화

        Args:
            dataname (str): 데이터셋의 이름
            scale (str): 데이터를 평가할 스케일의 이름
            path (str): 문장 데이터를 로드할 파일 경로
            field (str): 문장 컬럼의 필드명
        """

        self.dataname = dataname
        self.scale = scale
        self.path = path
        self.llm_labeled_path = llm_labeled_path
        self.ground_truth_path = ground_truth_path
        self.field = field
        self.scoring_method = scoring_method
        self.scaling_method = scaling_method
        self.adjustment_method = adjustment_method

        self.scatter_plot_cache = self._load_scatter_plot_cache(f'src/cache/{dataname}_{scale}_{adjustment_method}_scatter_plot_cache.pkl')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('`ScatterPlot` has been loaded')
        print(f'- Scoring method: {self.scoring_method}')
        print(f'- Scaling_method: {self.scaling_method}')
        print(f'- Adjustment method: {self.adjustment_method}')

        if self.adjustment_method == 'bertscore':
            self.scorer = BERTScorer('bert-base-uncased', lang='en')

        # Concept 목록
        self.concepts = CONCEPTS[scale]
        # Concept 라벨의 상대위치
        self.vertices = self._generate_vertices(len(self.concepts))
        
        # Concept별 긍부정문 few-shot 예시
        self.shots = {
            'true': self._load_shots(f'src/shots/{scale}/true'),
            'false': self._load_shots(f'src/shots/{scale}/false'),
        }

        # 캐시 존재하지 않을 시
        if path not in self.scatter_plot_cache:
            # BERT 토크나이저 & 모델
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True).to(self.device)
            
            # Concept별 긍부정문 키 임베딩 행렬
            self.key_embeddings = {
                'true': {
                    concept: np.vstack([self._get_embedding(desc.lower()) for desc in shots])
                    for concept, shots in self.shots['true'].items()
                },
                'false': {
                    concept: np.vstack([self._get_embedding(desc.lower()) for desc in shots])
                    for concept, shots in self.shots['false'].items()
                },
            }

    def _load_shots(self, directory: str) -> dict[str, list[str]]:
        """
        사전 정의된 OPRA concept의 few-shot 예시 로드

        Args:
            directory (str): 예시 파일이 저장된 디렉토리 경로

        Returns:
            dict[str, list[str]]: 파일 이름(확장자 제외)을 키로 하고 파일 내용의 각 행의 리스트를 값으로 하는 딕셔너리
        """

        shots = {}
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)

                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()

                concept = os.path.splitext(filename)[0]
                shots[concept] = content.splitlines()
        return shots
    
    def _load_scatter_plot_cache(self, filename: str) -> dict[str, tuple]:
        """
        캐시 파일에서 이전에 저장된 Scatter Plot 데이터를 로드

        Args:
            filename (str): 캐시 파일 이름

        Returns:
            dict[str, tuple]: 캐시된 Scatter Plot 데이터
        """

        if not USE_CACHE:
            return {}

        self.scatter_plot_cache_filename = filename
        # 캐시 파일 존재 시 로드
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                return pickle.load(file)
        return {}

    def _save_scatter_plot_cache(self) -> None:
        """
        현재 Scatter Plot 캐시를 파일에 저장
        """

        if not USE_CACHE:
            return

        with open(self.scatter_plot_cache_filename, 'wb') as file:
            pickle.dump(self.scatter_plot_cache, file)
    
    def _load_labeled_data(self, filename: str, volume: int) -> list[dict[str, float]]:
        """
        라벨링된 데이터 로드

        Args:
            filename (str): 라벨링된 데이터 파일 이름

        Returns:
            list[dict[str, float]]: 라벨링된 데이터
        """

        labeled = [{concept: 'N/A' for concept in self.concepts} for _ in range(volume)]
        labeled_df = pd.DataFrame(labeled)

        # 라벨링된 파일 존재 시 로드
        if filename is None or not os.path.exists(filename):
            return labeled
        df = pd.read_csv(filename)

        # # Concept 컬럼들이 소문자로 존재할 경우
        # if "Trust" not in df.columns and ("trust" in df.columns or "llm_trust" in df.columns):
        #     opra_column_names = {
        #         "trust": "Trust",
        #         "commitment": "Commitment",
        #         "control_mutuality": "Control Mutuality",
        #         "satisfaction": "Satisfaction",
        #         "llm_trust": "Trust",
        #         "llm_commitment": "Commitment",
        #         "llm_control_mutuality": "Control Mutuality",
        #         "llm_control mutuality": "Control Mutuality",
        #         "llm_satisfaction": "Satisfaction",
        #     }
        #     df = df.rename(columns=opra_column_names)

        # Concept 컬럼 이름 표준화
        df = df.rename(columns=CONCEPT_COLUMN_RENAME[self.scale])

        # Concept 컬럼이 없는 경우
        for column in self.concepts:
            if column not in df.columns:
                return labeled
        
        # Concept 컬럼만 선택하여 dict 타입으로 변환
        df = df.fillna('N/A')
        labeled_df.loc[:df.shape[0]-1, self.concepts] = df[self.concepts].values
        
        print(f'Labeled data `{filename}` has been loaded.')
        return labeled_df.to_dict(orient='records')

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        주어진 텍스트에 대한 BERT 임베딩 계산

        Args:
            text (str): 임베딩을 계산할 텍스트

        Returns:
            np.ndarray: 텍스트의 BERT 임베딩 벡터
        """

        #inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]

        # 문장의 CLS 토큰 임베딩 사용
        embedding = last_hidden_states[:, 0, :].detach().cpu().numpy()
        
        return embedding

    def _get_attention_embedding(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """
        주어진 쿼리에 대해 키들과의 어텐션 가중치를 사용하여 가중평균된 임베딩 계산

        Args:
            query (np.ndarray): 쿼리 벡터
            keys (np.ndarray): 키 벡터들의 배열

        Returns:
            tuple[np.ndarray, np.ndarray]
                - weighted_embedding (np.ndarray): 어텐션 가중치에 의해 가중된 임베딩
                - attention_wegiths (np.ndarray): 어텐션 가중치 배열
        """

        if query.ndim == 1:
            query = query[np.newaxis, :]

        # GPU로 이동
        query = torch.tensor(query, device=self.device)
        keys = torch.tensor(keys, device=self.device)
        
        # 쿼리, 키 벡터 사이의 dot product
        attention_scores = torch.matmul(query, keys.T)

        # Temperature scaling으로 softer distribution
        # temperature = 0.5
        # attention_scores = attention_scores / temperature

        # 어텐션 스코어에서 최대값 빼서 오버플로 방지
        # 이거 해도 softmax 결과에는 영향 없음
        max_scores = torch.max(attention_scores, dim=1, keepdim=True).values
        attention_scores -= max_scores

        # Softmax
        exp_scores = torch.exp(attention_scores)
        attention_weights = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
        # attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # attention_weights = (exp_scores * 2.0 / np.sum(exp_scores, axis=1, keepdims=True)) - 1.0

        # Weights 적용하여 새로운 임베딩
        weighted_embedding = torch.sum(attention_weights.unsqueeze(-1) * keys, dim=1)

        return weighted_embedding.cpu().numpy(), attention_weights.cpu().numpy()
    
    def _fit_concepts(self, positions_tsne: np.ndarray, query_embeddings: np.ndarray, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        t-SNE로 차원 축소된 임베딩을 concept 점수에 따라 조정.
        """
        # 초기 위치를 더 크게 스케일링 (중력 효과를 더 잘 보기 위해)
        positions = positions_tsne.copy()
        
        # t-SNE 결과를 적당한 크기로 스케일링
        max_distance = np.max(np.linalg.norm(positions, axis=1))
        if max_distance > 0:
            scale_factor = 0.6 / max_distance  # 0.8에서 0.6으로 줄여서 더 넓게 분포
            positions *= scale_factor
        
        concept_score = np.zeros((len(query_embeddings), len(self.concepts)))
        
        # concept_score 계산 (기존과 동일)
        if self.scoring_method == 'coc':
            alpha = 5.0  # softmax scale parameter (Guo et al., 2017; Hendrycks & Gimpel, 2017)
            for i, query_embedding in tqdm(enumerate(query_embeddings), total=len(query_embeddings), desc='Calculating concept scores'):
                for j, concept_name in enumerate(self.concepts):
                    _, true_attention = self._get_attention_embedding(
                        query_embedding, 
                        self.key_embeddings['true'][concept_name]
                    )
                    _, false_attention = self._get_attention_embedding(
                        query_embedding, 
                        self.key_embeddings['false'][concept_name]
                    )

                    # Use mean of max attention for each shot as the representative score
                    true = float(np.mean(np.max(true_attention, axis=1)))
                    false = float(np.mean(np.max(false_attention, axis=1)))

                    # Margin/softmax-based score (see Guo et al., 2017; Hendrycks & Gimpel, 2017)
                    exp_true = np.exp(alpha * true)
                    exp_false = np.exp(alpha * false)
                    prob_true = exp_true / (exp_true + exp_false)
                    concept_score[i][j] = prob_true
        elif self.scoring_method == 'bertscore':
            # BERTScore 방식
            for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc='Calculating concept scores'):
                for j, concept_name in enumerate(self.concepts):
                    true_shots = ' '.join(self.shots['true'][concept_name])
                    false_shots = ' '.join(self.shots['false'][concept_name])
                    
                    _, _, true_score = self.scorer.score([sentence], [true_shots])
                    _, _, false_score = self.scorer.score([sentence], [false_shots])

                    total_score = true_score + false_score
                    if total_score > 0:
                        concept_score[i][j] = true_score / total_score
                    else:
                        concept_score[i][j] = 0.5
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

        # 스케일링
        if self.scaling_method == 'minmax':
            scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # 0~1 대신 0.1~0.9로 여유 두기
            for j in range(len(self.concepts)):
                concept_score[:, j] = scaler.fit_transform(
                    concept_score[:, j].reshape(-1, 1)
                ).flatten()
        elif self.scaling_method == 'standard':
            scaler = StandardScaler()
            concept_score = scaler.fit_transform(concept_score)
            # 표준화 후 0~1 범위로 재조정
            concept_score = (concept_score - concept_score.min()) / (concept_score.max() - concept_score.min())
        
        # 위치 조정
        if self.adjustment_method == 'coc':
            positions = self._adjust_positions_coc(positions, concept_score)
        elif self.adjustment_method == 'gravity':
            positions = self._adjust_positions_gravity_constrained(positions, concept_score)
        # else:
        #     raise ValueError(f"Unknown adjusting method: {self.adjustment_method}")
        
        return positions, concept_score

    def _adjust_positions_coc(self, positions: np.ndarray, concept_score: np.ndarray) -> np.ndarray:
        """
        Certainty of Clues (COC) 방식으로 위치 조정.
        각 concept에 대한 점수를 기반으로 해당 방향으로 선형적으로 이동시킴.
        
        Args:
            positions (np.ndarray): 조정할 위치 배열
            concept_score (np.ndarray): concept 점수 배열
            
        Returns:
            np.ndarray: 조정된 위치 배열
        """
        positions = positions.copy()
        rank_multiplier = np.zeros_like(concept_score)
        
        # 각 concept별 intensity에 대해 순위 계산
        for j in range(len(self.concepts)):
            scores = concept_score[:, j]
            ranks = scores.argsort().argsort() + 1  # 순위 계산
            rank_multiplier[:, j] = ranks
        
        # positions에 순위를 적용하여 위치 조정
        for i in range(len(positions)):
            for j in range(len(self.concepts)):
                concept_vector = self.vertices[j][0] \
                    if concept_score[i][j] >= 0.5 \
                    else self.vertices[j][1]
                positions[i] += rank_multiplier[i][j] * concept_vector
        
        return positions

    def _adjust_positions_gravity(self, positions: np.ndarray, concept_score: np.ndarray) -> np.ndarray:
        """
        중력 모델을 사용하여 위치를 조정.
        각 concept는 중력점으로 작용하며, concept_score가 높을수록 더 강한 중력을 가짐.
        
        Args:
            positions (np.ndarray): 조정할 위치 배열 (t-SNE 결과)
            concept_score (np.ndarray): concept 점수 배열 (0.0-1.0)
        
        Returns:
            np.ndarray: 조정된 위치 배열
        """
        positions = positions.copy()
        
        # 중력 관련 상수들
        G = 1.0  # 중력 상수
        max_force = 0.5  # 최대 힘 제한
        min_distance = 0.01  # 최소 거리 (0으로 나누기 방지)
        
        # 각 점에 대해 중력 적용
        for i in tqdm(range(len(positions)), desc='Applying gravity'):
            total_force = np.zeros(2, dtype=float)
            
            for j in range(len(self.concepts)):
                # concept_score를 기반으로 질량과 방향 결정
                score = concept_score[i][j]
                
                # 0.5를 기준으로 true/false vertex 선택
                if score >= 0.5:
                    # True vertex로 끌림
                    vertex = self.vertices[j][0]  # true vertex
                    mass = (score - 0.5) * 2  # 0.0 ~ 1.0으로 정규화
                else:
                    # False vertex로 끌림
                    vertex = self.vertices[j][1]  # false vertex
                    mass = (0.5 - score) * 2  # 0.0 ~ 1.0으로 정규화
                
                # 현재 위치에서 vertex까지의 벡터
                displacement = np.array(vertex) - positions[i]
                distance = np.linalg.norm(displacement)
                
                # 거리가 너무 가까우면 건너뛰기
                if distance < min_distance:
                    continue
                
                # 정규화된 방향 벡터
                direction = displacement / distance
                
                # 중력 계산: F = G * mass / distance^2
                # 거리가 가까울수록 강한 힘, 질량이 클수록 강한 힘
                force_magnitude = G * mass / (distance ** 2)
                
                # 최대 힘 제한 (발산 방지)
                force_magnitude = min(force_magnitude, max_force)
                
                # 전체 힘에 추가
                total_force += direction * force_magnitude
            
            # 위치 업데이트 (작은 스텝으로 이동)
            step_size = 0.1  # 이동 스텝 크기
            positions[i] += total_force * step_size
        
        return positions
    
    def _adjust_positions_gravity_iterative(self, positions: np.ndarray, concept_score: np.ndarray) -> np.ndarray:
        """
        반복적 중력 시뮬레이션을 사용하여 위치를 조정.
        여러 번의 작은 스텝으로 안정적인 위치를 찾음.
        
        Args:
            positions (np.ndarray): 조정할 위치 배열
            concept_score (np.ndarray): concept 점수 배열
        
        Returns:
            np.ndarray: 조정된 위치 배열
        """
        positions = positions.copy()
        
        # 시뮬레이션 파라미터
        G = 0.5  # 중력 상수
        damping = 0.9  # 감쇠 계수 (진동 방지)
        max_iterations = 50  # 최대 반복 횟수
        convergence_threshold = 1e-4  # 수렴 임계값
        step_size = 0.05  # 이동 스텝 크기
        
        # 속도 초기화 (관성 효과)
        velocities = np.zeros_like(positions)
        
        for iteration in tqdm(range(max_iterations), desc='Gravity simulation'):
            forces = np.zeros_like(positions)
            
            # 각 점에 대해 중력 계산
            for i in range(len(positions)):
                total_force = np.zeros(2)
                
                for j in range(len(self.concepts)):
                    score = concept_score[i][j]
                    
                    # Vertex와 질량 결정
                    if score >= 0.5:
                        vertex = np.array(self.vertices[j][0])
                        mass = (score - 0.5) * 2
                    else:
                        vertex = np.array(self.vertices[j][1])
                        mass = (0.5 - score) * 2
                    
                    # 중력 계산
                    displacement = vertex - positions[i]
                    distance = np.linalg.norm(displacement)
                    
                    if distance > 1e-6:  # 0으로 나누기 방지
                        direction = displacement / distance
                        # 거리 제곱 대신 선형 관계 사용
                        force_magnitude = G * mass / (1 + distance)
                        total_force += direction * force_magnitude
                
                forces[i] = total_force
            
            # 속도 업데이트 (관성 적용)
            velocities = velocities * damping + forces * step_size
            
            # 위치 업데이트
            old_positions = positions.copy()
            positions += velocities
            
            # 수렴 확인
            position_change = np.linalg.norm(positions - old_positions)
            if position_change < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return positions
    
    def _calculate_concept_scores_improved(self, query_embeddings: np.ndarray, sentences: list[str]) -> np.ndarray:
        """
        개선된 concept score 계산
        
        Args:
            query_embeddings (np.ndarray): 쿼리 임베딩 배열
            sentences (list[str]): 문장 배열
            
        Returns:
            np.ndarray: concept 점수 배열 (0.0-1.0)
        """
        concept_score = np.zeros((len(query_embeddings), len(self.concepts)))
        
        for i, query_embedding in tqdm(enumerate(query_embeddings), total=len(query_embeddings), desc='Calculating concept scores'):
            for j, concept_name in enumerate(self.concepts):
                if self.scoring_method == 'coc':
                    # True/False attention 계산
                    _, true_attention = self._get_attention_embedding(
                        query_embedding, 
                        self.key_embeddings['true'][concept_name]
                    )
                    _, false_attention = self._get_attention_embedding(
                        query_embedding, 
                        self.key_embeddings['false'][concept_name]
                    )
                    
                    # 더 안정적인 점수 계산
                    true_weight = np.mean(np.max(true_attention, axis=1))
                    false_weight = np.mean(np.max(false_attention, axis=1))
                    
                    # 점수 정규화 (0.0-1.0)
                    total_weight = true_weight + false_weight
                    if total_weight > 0:
                        concept_score[i][j] = true_weight / total_weight
                    else:
                        concept_score[i][j] = 0.5  # 중립
                        
                elif self.scoring_method == 'bertscore':
                    # BERTScore 방식
                    sentence = sentences[i]
                    true_shots = ' '.join(self.shots['true'][concept_name])
                    false_shots = ' '.join(self.shots['false'][concept_name])
                    
                    _, _, true_score = self.scorer.score([sentence], [true_shots])
                    _, _, false_score = self.scorer.score([sentence], [false_shots])
                    
                    total_score = true_score + false_score
                    if total_score > 0:
                        concept_score[i][j] = true_score / total_score
                    else:
                        concept_score[i][j] = 0.5
        
        return concept_score

    # def _fit_concepts(self, positions_tsne: np.ndarray, query_embeddings: np.ndarray) -> np.ndarray:
    #     """
    #     긍정적 및 부정적 키 임베딩에 대한 어텐션 가중치를 기반으로 쿼리 임베딩의 2차원 공간 내 위치를 조정.
    #     이 메서드는 t-SNE를 통해 차원 축소된 임베딩을 긍정적 또는 부정적 개념에 대한 가중치에 따라 선형 변환함.

    #     Args:
    #         positions_tsne (np.ndarray): t-SNE를 사용하여 차원 축소된 임베딩의 2차원 위치 배열.
    #         query_embeddings (np.ndarray): 쿼리들의 임베딩 배열.

    #     Returns:
    #         tuple[np.ndarray, np.ndarray]: 2개의 값이 담긴 튜플 반환.
    #             - positions: 긍정적 및 부정적 concept의 영향을 반영하여 조정된 각 쿼리의 2차원 공간 내 위치 배열.
    #             - concept_score: 쿼리 임베딩의 각 concept에 대한 점수 (0.0-1.0).
    #     """
    
    #     positions = positions_tsne
    #     concept_score = np.zeros((len(query_embeddings), len(self.concepts)))
    #     rank_multiplier = np.zeros_like(concept_score)

    #     # concept_score 계산
    #     for i, query_embedding in tqdm(enumerate(query_embeddings), total=len(query_embeddings), desc='Fitting positions in scatter plot'):
    #         # 각 concept별 영향을 반영하여 위치 계산
    #         for j in range(len(self.concepts)):  # 모든 concept에 대해 반복
    #             concept_name = self.concepts[j]

    #             # Concept의 긍부정 예시에 대한 attention weight 계산
    #             _, positive_attention = self._get_attention_embedding(query_embedding, self.key_embeddings['positive'][concept_name])
    #             _, negative_attention = self._get_attention_embedding(query_embedding, self.key_embeddings['negative'][concept_name])

    #             # Concept의 긍부정 예시에 대한 intensity weight 계산
    #             positive_weight = np.max(np.median(positive_attention, axis=1))
    #             negative_weight = np.max(np.median(negative_attention, axis=1))

    #             # concept에 대한 sentence의 intensity 계산
    #             concept_score[i][j] = 0.5 + positive_weight / 2 \
    #                 if positive_weight > negative_weight \
    #                 else 0.5 - negative_weight / 2
        
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     for j in range(len(self.concepts)):
    #         # concept_score를 각 concept별로 MinMax 스케일링
    #         concept_score[:, j] = scaler.fit_transform(concept_score[:, j].reshape(-1, 1)).flatten()

    #         # 각 concept별 intensity에 대해 순위 계산
    #         scores = concept_score[:, j]
    #         ranks = scores.argsort().argsort() + 1  # 순위 계산; 더 낮은 순위는 더 높은 intensity를 의미
    #         rank_multiplier[:, j] = ranks

    #     # positions_tsne에 순위를 적용하여 위치 조정
    #     for i in range(len(query_embeddings)):
    #         for j in range(len(self.concepts)):
    #             concept_vector = self.vertices[j][0] \
    #                 if concept_score[i][j] >= 0.5 \
    #                 else self.vertices[j][1]
    #             # 순위(rank_multiplier)를 이용해 위치 조정
    #             positions[i] += rank_multiplier[i][j] * concept_vector

    #     return positions, concept_score

    def _sentence_positions(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        주어진 문장들에 대한 2차원 공간 내 위치 계산

        Args:
            sentences (list[str]): 위치를 계산할 문장들의 리스트

        Returns:
            tuple[np.ndarray, np.ndarray]: 2개의 값이 담긴 튜플 반환
                - positions_tsne: t-SNE를 사용하여 차원 축소한 문장 임베딩의 2차원 위치 배열
                - positions: t-SNE 이후 concept와의 어텐션을 기반으로 이동한 문장 임베딩의 2차원 위치 배열
                - concept_score: 문장들의 각 concept에 대한 점수 (0.0-1.0).
        """

        # 입력 문장의 쿼리 임베딩 행렬
        query_embeddings = np.vstack([self._get_embedding(sent.lower()) for sent in tqdm(sentences, desc='Calculating query embeddings')])

        # t-SNE를 사용하여 문장 임베딩을 차원 축소한 위치 계산
        perplexity_value = min(30, max(5, len(sentences) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
        positions_tsne = tsne.fit_transform(query_embeddings)

        # Concept와의 어텐션을 기반으로 입력 문장의 2차원 위치 선형 이동 & 점수 계산
        positions, concept_score = self._fit_concepts(positions_tsne, query_embeddings, sentences)

        return positions_tsne, positions, concept_score

    def _load_position_data(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        CSV 파일로부터 문장 데이터를 로드하고, 각 문장에 대해 BERT 기반의 임베딩을 계산하여
        2차원 공간 내에서의 위치와 t-SNE를 통해 차원 축소한 후의 위치를 계산

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]: 3개의 값이 담긴 튜플 반환
                - sentences: 로드된 문장들의 리스트
                - positions_tsne: t-SNE를 사용하여 차원 축소한 문장 임베딩의 2차원 위치 배열
                - positions: t-SNE 이후 concept와의 어텐션을 기반으로 이동한 문장 임베딩의 2차원 위치 배열
                - concept_score: 문장 임베딩의 각 concept에 대한 점수 (0.0-1.0).
        """

        df = pd.read_csv(self.path)
        sentences = df[self.field].tolist()
        positions_tsne, positions, concept_score = self._sentence_positions(sentences)

        return sentences, positions_tsne, positions, concept_score

    def _transform_coordinates_to_octagon(self, embeddings: np.ndarray) -> np.ndarray:
        """
        주어진 임베딩 좌표를 변환하여 8각형 내부에 존재하도록 범위에 맞게 조정

        Args:
            embeddings (np.ndarray): 2차원 좌표를 포함하는 numpy 배열. 각 행은 하나의 포인트를 나타냄.

        Returns:
            np.ndarray: 변환된 좌표를 포함하는 numpy 배열. 입력 배열과 같은 형태를 가짐.
        """

        embeddings[:, 1] = -embeddings[:, 1]
        center = np.array([0, 0])
        angles = np.degrees(np.arctan2(embeddings[:, 1], embeddings[:, 0]))

        distances = np.linalg.norm(embeddings, axis=1)

        # 8각형 변에 해당하는 각도 범위 정의
        angle_ranges = [
            (22.5, 67.5), (112.5, 157.5),
            (-67.5, -22.5), (-157.5, -112.5)
        ]

        for i, (angle, distance) in enumerate(zip(angles, distances)):
            # 각도 범위에 따라 처리
            for angle_range in angle_ranges:
                if angle_range[0] < angle <= angle_range[1]:
                    scale_factor = np.sqrt(.5)  # 거리 스케일 조정 비율
                    direction_vector = (embeddings[i] - center) / distance  # 단위 방향 벡터
                    new_distance = distance * scale_factor  # 조정된 거리
                    embeddings[i] = center + direction_vector * new_distance  # 새 위치 계산
                    break  # 현재 점에 대한 처리가 완료되면 다음 점으로 이동

        return (embeddings - center)
    
    def _transform_coordinates_to_circle(self, embeddings: np.ndarray) -> np.ndarray:
        """
        주어진 임베딩 좌표를 원형으로 스케일링하여, 모든 포인트가 주어진 반지름을 가진 원 위에 위치하도록 조정

        Args:
            embeddings (np.ndarray): 2차원 좌표를 포함하는 numpy 배열. 각 행은 하나의 포인트를 나타냄.
        
        Returns:
            np.ndarray: 원형으로 스케일링된 좌표를 포함하는 numpy 배열.
        """

        embeddings[:, 1] = -embeddings[:, 1]
        center = np.array([0, 0])
        
        # 반지름: 중심으로부터 가장 먼 포인트까지의 거리
        distances = np.linalg.norm(embeddings - center, axis=1)
        radius = distances.max()
        
        # 원의 바깥에 있는 포인트를 원 위로 스케일링
        scale_factors = np.where(distances > radius, 1, distances / radius)
    
        # 각 점을 스케일링
        scaled_embeddings = center + (embeddings - center) * scale_factors[:, np.newaxis]
        
        return scaled_embeddings
    
    def _calculate_similarity(self, sentences: list[str]) -> np.ndarray:
        """
        문장들 간 코사인 유사도 계산

        Args:
            sentences (list[str]): 로드된 문장들의 리스트

        Returns:
            np.ndarray: 유사도 행렬
        """

        print('Calculating similarity...')

        # 입력 문장의 임베딩 행렬
        embeddings = np.vstack([self._get_embedding(sent.lower()) for sent in sentences])

        # embeddings 배열에 대해 코사인 유사도 계산
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def get_scatter_plot_data(self) -> tuple[list[dict[str, float]], list[list]]:
        """
        저장된 문장 데이터를 로드하고, 각 문장에 대한 위치 정보를 변환하여 
        웹에서 시각화하기 적합한 형식으로 준비

        Returns:
            list[dict[str, float]]: scatter plot에 사용될 데이터를 포함하는 리스트.
                각 항목은 'source', 'sentence', 'x', 'y', 'title', 'content' 키를 가진 딕셔너리.
            list[list]: 유사도 행렬.
        """

        print('Preparing scatter plot data...')

        # 위치 정보는 캐시에서 불러오거나 새로 계산
        if self.path not in self.scatter_plot_cache:
            sentences, positions_tsne, positions, concept_score = self._load_position_data()
            similarity_matrix = self._calculate_similarity(sentences)
            
            self.scatter_plot_cache[self.path] = (sentences, positions_tsne, positions, concept_score, similarity_matrix)
            self._save_scatter_plot_cache()
        elif USE_CACHE:
            sentences, positions_tsne, positions, concept_score, similarity_matrix = self.scatter_plot_cache[self.path]

        # 웹에 시각화하기 위한 데이터 형식으로 변환
        positions_scaled = positions
        # scaler = MinMaxScaler(feature_range=(-1, 1)) # -1에서 1 사이로 스케일 조정
        # scaler.fit(positions)
        # positions_scaled = scaler.transform(positions)
        positions_scaled = list(self._transform_coordinates_to_circle(positions_scaled))

        # 이미 레이팅된 데이터 로드
        llm_labeled = self._load_labeled_data(self.llm_labeled_path, len(positions_scaled)) # LLM이 레이팅한 데이터
        ground_truth = self._load_labeled_data(self.ground_truth_path, len(positions_scaled)) # Ground-truth 데이터

        # 데이터 벡터 처리
        positions_scaled = np.round(positions_scaled, 4)
        concept_score = np.round(concept_score, 4)

        scatter_df = pd.DataFrame({
            # 'source': float(-1),
            # 'sentence': float(i),
            'x': positions_scaled[:, 0],
            'y': positions_scaled[:, 1],
            # 'title': 'No Duplicate',
            'content': sentences,
            'opra': [dict(zip(self.concepts, scores)) for scores in concept_score], # OPRA concept에 대한 attention (COC 점수)
            'opra_label_gt': ground_truth, # Ground-truth label
            'opra_label_llm': llm_labeled, # LLM의 label
        })

        # 유사도 행렬 양자화
        similarity_matrix = (similarity_matrix * 255).astype(np.uint8)

        return scatter_df.to_dict(orient='records'), similarity_matrix

    def _generate_vertices(self, num_concepts: int) -> np.ndarray:
        """
        주어진 차원 수에 대한 vertices 생성

        Args:
            num_concepts (int): Concept의 수
        
        Returns:
            np.ndarray: 각 concept의 true/false vertices [true vertex, false vertex]
        """
        
        # 시계 방향 각도 설정 (단위: 라디안)
        hours = np.linspace(0, 6, num_concepts, endpoint=False) # 시계에서 각 concept의 시간 위치
        angles_true = (hours * 30) * (np.pi / 180) # 1시간 = 30도
        angles_false = angles_true + np.pi # False의 방향은 180도 반대

        vertices = []
        for angle_true, angle_false in zip(angles_true, angles_false):
            true = [np.sin(angle_true), np.cos(angle_true)]
            false = [np.sin(angle_false), np.cos(angle_false)]
            vertices.append([true, false])
        return np.array(vertices)

    def _get_polygon_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        다각형의 각 변(edge)을 정의
        각 변은 concept를 나타냄
        
        Returns:
            list[tuple[np.ndarray, np.ndarray]]: 각 변의 시작점과 끝점 좌표
        """
        num_concepts = len(self.concepts)
        edges = []
        
        # 다각형의 꼭짓점들 계산 (단위원 위의 정다각형)
        angles = np.linspace(0, 2 * np.pi, num_concepts + 1)[:-1]  # 마지막 점 제외
        vertices = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
        
        # 각 변 정의 (연속된 두 꼭짓점을 연결)
        for i in range(num_concepts):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % num_concepts]
            edges.append((start_vertex, end_vertex))
        
        return edges

    def _point_to_edge_distance_and_projection(self, point: np.ndarray, edge: tuple[np.ndarray, np.ndarray]) -> tuple[float, np.ndarray]:
        """
        점에서 변까지의 최단거리와 투영점을 계산
        
        Args:
            point (np.ndarray): 점의 좌표
            edge (tuple[np.ndarray, np.ndarray]): 변의 시작점과 끝점
        
        Returns:
            tuple[float, np.ndarray]: (거리, 투영점 좌표)
        """
        start, end = edge
        edge_vector = end - start
        point_vector = point - start
        
        # 변의 길이
        edge_length_squared = np.dot(edge_vector, edge_vector)
        
        if edge_length_squared == 0:
            # 변의 길이가 0인 경우 (시작점과 끝점이 같음)
            return np.linalg.norm(point - start), start
        
        # 점을 변에 투영했을 때의 매개변수 t
        t = np.dot(point_vector, edge_vector) / edge_length_squared
        
        # t를 [0, 1] 범위로 제한 (변의 범위 내)
        t = max(0, min(1, t))
        
        # 투영점 계산
        projection = start + t * edge_vector
        
        # 거리 계산
        distance = np.linalg.norm(point - projection)
        
        return distance, projection

    def _is_point_inside_polygon(self, point: np.ndarray) -> bool:
        """
        점이 다각형 내부에 있는지 확인
        
        Args:
            point (np.ndarray): 확인할 점의 좌표
        
        Returns:
            bool: 점이 다각형 내부에 있으면 True
        """
        edges = self._get_polygon_edges()
        
        # Ray casting algorithm
        x, y = point
        inside = False
        
        for start, end in edges:
            x1, y1 = start
            x2, y2 = end
            
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        
        return inside

    def _get_polygon_boundary_constraint(self, point: np.ndarray) -> np.ndarray:
        """
        점이 다각형 경계를 벗어나면 경계 내부로 투영
        
        Args:
            point (np.ndarray): 확인할 점
        
        Returns:
            np.ndarray: 경계 내부로 조정된 점
        """
        # 원점에서의 거리
        distance = np.linalg.norm(point)
        
        if distance <= 1.0:
            return point
        
        # 각도 계산
        angle = np.arctan2(point[1], point[0])
        if angle < 0:
            angle += 2 * np.pi
        
        # 해당 각도에서 다각형 경계까지의 거리 계산
        num_concepts = len(self.concepts)
        sector_angle = 2 * np.pi / num_concepts
        
        # 어느 섹터에 속하는지 찾기
        sector_index = int(angle / sector_angle) % num_concepts
        
        # 해당 섹터의 두 vertex 사이의 경계선에 투영
        vertex1 = np.array(self.vertices[sector_index][0])  # true vertex
        vertex2 = np.array(self.vertices[(sector_index + 1) % num_concepts][0])  # 다음 true vertex
        
        # 두 vertex를 잇는 직선에 투영
        edge_vector = vertex2 - vertex1
        point_vector = point - vertex1
        
        if np.linalg.norm(edge_vector) > 1e-6:
            t = np.dot(point_vector, edge_vector) / np.dot(edge_vector, edge_vector)
            t = max(0, min(1, t))  # [0, 1] 범위로 제한
            projection = vertex1 + t * edge_vector
        else:
            projection = vertex1
        
        return projection

    def _adjust_positions_gravity_constrained(self, positions: np.ndarray, concept_score: np.ndarray) -> np.ndarray:
        """
        다각형 경계 내에서 concept별 vertex로 끌어당기는 중력을 적용
        
        Args:
            positions (np.ndarray): 조정할 위치 배열
            concept_score (np.ndarray): concept 점수 배열
        
        Returns:
            np.ndarray: 조정된 위치 배열
        """
        positions = positions.copy()
        
        # 디버깅: concept별 높은 점수를 가진 점들 확인
        print("Debugging concept scores:")
        for j, concept_name in enumerate(self.concepts):
            high_score_indices = np.where(concept_score[:, j] > 0.7)[0]
            print(f"{concept_name}: {len(high_score_indices)} points with score > 0.7")
            if len(high_score_indices) > 0:
                print(f"  Max score: {np.max(concept_score[:, j]):.3f}")
                print(f"  Target vertex (true): {self.vertices[j][0]}")
                print(f"  Target vertex (false): {self.vertices[j][1]}")
        
        # 시뮬레이션 파라미터
        max_iterations = 200
        step_size = 0.1  # 더 큰 스텝
        damping = 0.8
        convergence_threshold = 1e-4
        base_attraction = 2.0  # 기본 끌어당김 강도 증가
        
        # 속도 초기화
        velocities = np.zeros_like(positions)
        
        for iteration in tqdm(range(max_iterations), desc='Constrained gravity simulation'):
            forces = np.zeros_like(positions)
            
            for i in range(len(positions)):
                total_force = np.zeros(2)
                
                # 각 concept에 대해 해당 vertex로의 끌어당김 계산
                for j in range(len(self.concepts)):
                    score = concept_score[i][j]
                    
                    # 더 명확한 임계값 설정
                    if score > 0.5:  # 0.5 이상일 때만 true로 끌어당김
                        target_vertex = np.array(self.vertices[j][0])
                        attraction_strength = (score - 0.5) * base_attraction
                    elif score < 0.5:  # 0.5 미만일 때만 false로 끌어당김
                        target_vertex = np.array(self.vertices[j][1])
                        attraction_strength = (0.5 - score) * base_attraction
                    else:
                        # 중간 점수는 끌어당기지 않음
                        continue
                    
                    # 중력 계산
                    displacement = target_vertex - positions[i]
                    distance = np.linalg.norm(displacement)
                    
                    if distance > 1e-6:
                        direction = displacement / distance
                        # 거리에 따른 힘 조정 (너무 가까우면 약하게, 적당한 거리에서 강하게)
                        if distance < 0.1:
                            force_magnitude = attraction_strength * distance * 10  # 가까우면 약하게
                        else:
                            force_magnitude = attraction_strength / (1 + distance * 0.5)  # 멀면 강하게
                        
                        total_force += direction * force_magnitude
                
                forces[i] = total_force
            
            # 속도 및 위치 업데이트
            velocities = velocities * damping + forces * step_size
            old_positions = positions.copy()
            new_positions = positions + velocities
            
            # 경계 제약 (단위원 내부로 제한)
            for i in range(len(new_positions)):
                distance_from_origin = np.linalg.norm(new_positions[i])
                
                if distance_from_origin <= 0.95:  # 여유를 두고 0.95로 제한
                    positions[i] = new_positions[i]
                else:
                    # 경계로 투영
                    positions[i] = new_positions[i] * (0.95 / distance_from_origin)
                    velocities[i] *= 0.3  # 경계에 닿으면 속도 감소
            
            # 수렴 확인
            if iteration % 10 == 0:
                position_change = np.linalg.norm(positions - old_positions)
                if position_change < convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        # 최종 결과 디버깅
        print("\nFinal position analysis:")
        for j, concept_name in enumerate(self.concepts):
            high_score_indices = np.where(concept_score[:, j] > 0.7)[0]
            if len(high_score_indices) > 0:
                target_vertex = np.array(self.vertices[j][0])
                avg_distance = np.mean([
                    np.linalg.norm(positions[i] - target_vertex) 
                    for i in high_score_indices
                ])
                print(f"{concept_name}: Average distance to target vertex: {avg_distance:.3f}")
        
        return positions
