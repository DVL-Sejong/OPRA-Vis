from flask import Flask, render_template, request, jsonify, redirect, Response
from src import *
import pandas as pd
import dotenv
import os
import json
import gzip
import argparse

CONFIG_FILE = 'config.json'

# 커맨드라인 파라미터
parser = argparse.ArgumentParser(description='Visual Analytics for Labeling Communication Behavioral Data')
parser.add_argument('--data', type=str, help='The name of data to load', required=True, choices=('amazon', 'local', 'imdb', 'jigsaw'))
parser.add_argument('--scale', type=str, help='The name of the measurement scale', required=True, choices=('opra', 'toxicity'))
parser.add_argument('--volume', type=int, help='The volume of data to load', required=False)
# parser.add_argument('--force_method', type=str, help='', required=False, choices=('coc', 'gravity'), default='coc')
parser.add_argument('--disable_word_cloud', action='store_true', help='Disable the word cloud functionality', default=False)
args = parser.parse_args()

dotenv.load_dotenv()
app = Flask(__name__)

@app.route('/')
def index() -> str:
    logger = setup_user_logger()
    logger.info("Index page accessed")
    return render_template('index.html',
        dataname=args.data,
        scale=args.scale,
        concepts=json.dumps(CONCEPTS[args.scale]),
    )

@app.route('/metadata', methods=['GET'])
def metadata() -> str:
    metadata = {
        'dataname': args.data,
        'scale': args.scale,
        'volume': args.volume,
        'concepts': CONCEPTS[args.scale],
        'concept_column_names': CONCEPT_COLUMN_NAMES[args.scale],
    }
    return json.dumps(metadata)

@app.route('/scatter_plot', methods=['GET'])
def scatter_plot() -> str:
    global scatter_plot_data

    # JSON 스트리밍
    def generate_chunk():
        for data_point in scatter_plot_data:
            yield json.dumps(data_point) + '\n'

    return Response(generate_chunk(), content_type='application/json')
    # return json.dumps({'data': scatter_plot_data})
    # 데이터 형식:
    # [{"source": float, "sentence": float, "x": float "y": float, "title": str "content": str,
    #   "opra": {"Trust": float, "Commitment": float, "Control Mutuality": float, "Satisfaction": float}},
    # ...]

@app.route('/similarity', methods=['GET'])
def similarity() -> str:
    global similarity_matrix

    # 바이너리 데이터 압축
    compressed = gzip.compress(similarity_matrix.tobytes())

    # 바이너리 스트리밍
    def generate_chunk():
        chunk_size = 4096
        for i in range(0, len(compressed), chunk_size):
            yield compressed[i:i+chunk_size]
    
    return Response(generate_chunk(), content_type='application/octet-stream')

@app.route('/decision', methods=['POST'])
def decision() -> str:
    request_data = request.get_json()
    concept: str = request_data['concept']
    content_id_list: list[int] = request_data['content_id_list']
    prompt: str | None = request_data['prompt']

    global scatter_plot_data
    contents = [data['content'] for data in scatter_plot_data]

    logger = setup_user_logger()
    if prompt is not None:
        logger.info("Perform LLM assessments")
        logger.info(f"\t- Contents IDs: {content_id_list[0]}")
        logger.info(f"\t- Contents: {contents[content_id_list[0]]}")
        logger.info(f"\t- Concept: {concept}")
        logger.info(f"\t- Prompt: {prompt}")
    else:
        logger.info("Perform LLM assessments")
        logger.info(f"\t- Contents IDs: {content_id_list[0]}")
        logger.info(f"\t- Contents: {contents[content_id_list[0]]}")
        logger.info(f"\t- Concept: {concept}")
        logger.info(f"\t- Prompt: (default)")

    global llm
    llm_data = llm.get_llm_data(concept, contents, content_id_list, 'carp_os', prompt=prompt)

    return json.dumps({'data': llm_data})

@app.route('/sentiment', methods=['GET'])
def sentiment() -> str:
    global word_cloud_data

    # JSON 스트리밍
    def generate_chunk():
        yield '{'
        for i, (key, value) in enumerate(word_cloud_data.items()):
            if i > 0:
                yield ','
            yield f'"{key}": {json.dumps(value)}'
        yield '}'

    return Response(generate_chunk(), content_type='application/json')
    # return json.dumps(word_cloud_data)

@app.route('/configs', methods=['GET', 'POST'])
def configs():
    if request.method == 'POST':
        # 설정 저장
        data = request.json
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f)
        return jsonify({'message': 'Configurations saved successfully'}), 200
    else:
        # 설정 불러오기
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            return jsonify(data), 200
        else:
            return jsonify({'message': 'No configurations found'}), 404

@app.route('/log', methods=['POST'])
def log():
    request_data = request.get_json()
    message: str = request_data['message']

    logger = setup_user_logger()
    logger.info(message)

    return jsonify({'message': 'Logged successfully'}), 200

@app.route('/sus', methods=['GET'])
def sus():
    return redirect('https://forms.gle/iVqU1xhzxurz1atN6')

def preload() -> None:
    def load_scatter_plot_data():
        # Scatter plot 데이터 로드
        global scatter_plot_data
        global similarity_matrix

        # 경로 처리
        base_dir = f'static/data/{args.data}'
        datapath = f'{base_dir}/{args.data}_labeled.csv' # 데이터 경로
        llmpath = f'{base_dir}/{args.data}_llm.csv' # LLM이 레이팅한 데이터 경로
        gtpath = f'{base_dir}/{args.data}_human.csv' # Ground-truth 데이터 경로
        if args.volume is not None:
            base_dir = f'static/data/{args.data}_{args.volume}'
            datapath = f'{base_dir}/{args.data}_{args.volume}_labeled.csv'
            llmpath = f'{base_dir}/{args.data}_llm.csv' # LLM이 레이팅한 데이터 경로
            gtpath = f'{base_dir}/{args.data}_human.csv' # Ground-truth 데이터 경로
        if not os.path.exists(datapath):
            raise FileNotFoundError(f'The file `{datapath}` was not found.')

        scatter_plot = ScatterPlot(
            dataname=args.data,
            scale=args.scale,
            path=datapath,
            llm_labeled_path=llmpath,
            ground_truth_path=gtpath,
            field=DATA_TEXT_FIELD[args.data],
            # force_method=args.force_method,
            # adjustment_method=None,
        )
        scatter_plot_data, similarity_matrix = scatter_plot.get_scatter_plot_data()
        print("Scatter plot data loaded")

    def load_word_cloud_data():
        # Word cloud 데이터 로드
        global word_cloud_data
        
        if args.disable_word_cloud:
            word_cloud_data = {}
            return

        # 경로 처리
        base_dir = f'static/data/{args.data}/sentiment'
        if args.volume is not None:
            base_dir = f'static/data/{args.data}_{args.volume}/sentiment'
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f'The directory `{base_dir}` was not found.')
        
        concepts = [concept.lower() for concept in CONCEPTS[args.scale]]
        clues = CLUE_LABELS
        sentiments = SENTIMENT_LABELS

        word_cloud_data = {}

        for sentiment in sentiments:
            for clue in clues:
                for concept in concepts:
                    data_key = f'{concept}_{clue}_{sentiment}'
                    file_path = f'{base_dir}/{data_key}.csv'
                    if not os.path.exists(file_path):
                        concept_underscore = concept.replace(' ', '_')
                        data_key = f'{concept_underscore}_{clue}_{sentiment}'
                        file_path = f'{base_dir}/{concept_underscore}_{clue}_{sentiment}.csv'

                    if os.path.exists(file_path):
                        # 상위 1000개(최대)의 데이터만 로드
                        df = pd.read_csv(file_path).head(1000).values.tolist()
                        word_cloud_data[data_key] = df
                    else:
                        # 파일이 없을 경우 빈 리스트로 처리
                        word_cloud_data[data_key] = []
                        print(f'File does not exist: {file_path}')

    def load_llm():
        # LLM decision making 모델 로드
        global llm
        gemma_token = os.environ.get('GEMMA_TOKEN')
        llm = LLM("google/gemma-7b", token=gemma_token)
    
    load_scatter_plot_data()
    load_word_cloud_data()
    load_llm()

if __name__ == '__main__':
    preload()

    # 환경 변수에서 SSL 인증서와 키 파일 경로 로드
    ssl_cert_path = os.environ.get('SSL_CERT_PATH')
    ssl_key_path = os.environ.get('SSL_KEY_PATH')

    # 환경 변수에서 host와 port 설정 로드
    host = os.environ.get('HOST')
    port = os.environ.get('PORT')
    
    # SSL 환경 변수 유무에 따라 HTTPS or HTTP로 Flask 애플리케이션 실행
    if ssl_cert_path and ssl_key_path:
        app.run(debug=False, host=host, port=port, threaded=True, ssl_context=(ssl_cert_path, ssl_key_path))
    else:
        app.run(debug=False, host=host, port=port, threaded=True)
