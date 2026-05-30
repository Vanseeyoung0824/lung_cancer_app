# 🫁 폐암 환자 군집 분석 시스템

AI(K-Means 클러스터링)를 활용하여 폐암 환자의 특성을 분석하고,  
어떤 군집(유형)에 속하는지 예측하는 Streamlit 웹 애플리케이션입니다.

## 군집 정의

| 군집 | 이름 | 설명 |
|------|------|------|
| 0번 | 매우 건강군 | 낮은 흡연량·음주량, 비교적 젊은 연령대 |
| 1번 | 위험군 | 높은 흡연량, 중장년층, 폐암 위험 가장 높음 |
| 2번 | 건강군 | 보통 수준의 흡연량·음주량, 중간 위험도 |

## 로컬 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/<your-username>/lung-cancer-cluster-app.git
cd lung-cancer-cluster-app

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 앱 실행
streamlit run app.py
```

## Streamlit Community Cloud 배포 방법

1. 이 저장소를 GitHub에 Push
2. [share.streamlit.io](https://share.streamlit.io) 접속 → "New app" 클릭
3. 저장소 선택 → Main file: `app.py` → Deploy!

## 기술 스택

- **Frontend / UI**: Streamlit
- **ML 모델**: scikit-learn KMeans
- **시각화**: Matplotlib
- **데이터 처리**: Pandas, NumPy
