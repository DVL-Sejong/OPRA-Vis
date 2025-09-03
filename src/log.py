import logging
from flask import request
from datetime import datetime
import os

def setup_user_logger():
    user_ip = request.remote_addr  # 사용자 IP 주소
    now = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"logs/{user_ip}_{now}.log"  # 로그 파일명에 IP 주소와 날짜 포함

    # 로그 저장 디렉토리 생성 (없을 경우)
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # 로거 설정
    logger = logging.getLogger(user_ip)  # 사용자 IP를 기반으로 한 로거 인스턴스 생성
    if not logger.handlers:  # 핸들러 중복 추가 방지
        logger.setLevel(logging.INFO)  # 로그 레벨 설정

        # 파일 핸들러 설정
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
