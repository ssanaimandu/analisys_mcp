# 1. 베이스 이미지 선택 (Python 3.10이 설치된 Alpine Linux)
FROM python:3.10-alpine

# --- 환경 변수 설정 ---
# .pyc 파일을 생성하지 않도록 설정
ENV PYTHONDONTWRITEBYTECODE 1
# Python 출력이 버퍼링 없이 즉시 터미널에 표시되도록 설정
ENV PYTHONUNBUFFERED 1

# 2. 필수 리눅스 유틸리티 및 빌드 도구 설치
#    - bash: Bash 셸
#    - procps-ng: ps 명령어 제공
#    - net-tools: netstat, ifconfig 명령어 제공
#    - tcpdump: tcpdump 유틸리티
#    - build-base, freetype-dev, libpng-dev, g++: matplotlib 등 Python 패키지 빌드에 필요한 의존성
RUN apk add --no-cache \
    bash \
    procps-ng \
    net-tools \
    tcpdump \
    build-base \
    freetype-dev \
    libpng-dev \
    g++

# 3. 작업 디렉터리 설정
WORKDIR /app

# 4. Python 라이브러리 설치 (소스 코드 복사 전 수행하여 레이어 캐싱 활용)
#    - 먼저 requirements.txt 파일만 복사
COPY requirements.txt .
#    - pip를 최신 버전으로 업그레이드하고, requirements.txt에 명시된 라이브러리 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY analisys_llm.py .

# 6. 컨테이너 실행 시 기본 명령어 설정
#    - FastMCP가 stdio 모드로 실행되도록 설정
CMD ["python", "analisys_llm.py"]
