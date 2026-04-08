#!/bin/bash
echo "SSU CQI 분석 플랫폼 시작 중..."

# 한글 폰트 설치 (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y fonts-nanum 2>/dev/null || true
fi

# 패키지 설치
pip install -r requirements.txt

# .env 파일 없으면 예시에서 복사
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env 파일이 생성되었습니다. ANTHROPIC_API_KEY를 입력하세요."
fi

# 앱 실행
streamlit run app.py
