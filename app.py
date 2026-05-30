import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, urllib.request

# ── 한글 폰트 설정 (Streamlit Cloud 대응) ──────────────────────
def set_korean_font():
    font_path = "/tmp/NanumGothic.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(
            "https://github.com/googlefonts/nanum/raw/main/src/NanumGothic/NanumGothic-Regular.ttf",
            font_path,
        )
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지

set_korean_font()

# ── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석 시스템",
    page_icon="🫁",
    layout="centered",
)

# ── CSS 스타일 ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .sub-desc {
        text-align: center;
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .result-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        font-size: 0.95rem;
        color: #166534;
        margin-bottom: 0.8rem;
    }

    .result-box.warning {
        background-color: #fff7ed;
        border-left-color: #f97316;
        color: #9a3412;
    }

    .result-box.danger {
        background-color: #fef2f2;
        border-left-color: #ef4444;
        color: #991b1b;
    }

    .legend-text {
        font-size: 0.88rem;
        color: #444;
        margin-top: 0.3rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── 샘플 데이터 생성 (KMeans 학습용) ────────────────────────────
@st.cache_data
def generate_sample_data():
    np.random.seed(42)

    # 군집 0: 매우 건강군 — 낮은 흡연량, 낮은 음주량, 젊은 나이
    c0 = pd.DataFrame({
        "나이": np.random.normal(35, 5, 30).clip(20, 55),
        "흡연량": np.random.normal(3, 2, 30).clip(0, 8),
        "음주량": np.random.normal(1, 1, 30).clip(0, 4),
    })

    # 군집 1: 위험군 — 높은 흡연량, 중간 음주량, 중장년
    c1 = pd.DataFrame({
        "나이": np.random.normal(60, 7, 30).clip(40, 80),
        "흡연량": np.random.normal(20, 5, 30).clip(10, 35),
        "음주량": np.random.normal(5, 2, 30).clip(2, 10),
    })

    # 군집 2: 건강군 — 보통 흡연량, 낮은 음주량, 중간 나이
    c2 = pd.DataFrame({
        "나이": np.random.normal(48, 6, 30).clip(30, 65),
        "흡연량": np.random.normal(10, 3, 30).clip(5, 20),
        "음주량": np.random.normal(3, 1, 30).clip(1, 6),
    })

    df = pd.concat([c0, c1, c2], ignore_index=True)
    return df


@st.cache_resource
def train_model(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["나이", "흡연량", "음주량"]])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # 군집 레이블 재정렬: 평균 흡연량 기준으로 0=매우 건강, 1=위험, 2=건강
    cluster_means = pd.DataFrame({
        "cluster": range(3),
        "mean_smoking": [df[labels == i]["흡연량"].mean() for i in range(3)],
    }).sort_values("mean_smoking")

    # 흡연량 낮은 순: 0(매우건강), 2(건강), 1(위험)
    rank_map = {
        cluster_means.iloc[0]["cluster"]: 0,
        cluster_means.iloc[1]["cluster"]: 2,
        cluster_means.iloc[2]["cluster"]: 1,
    }
    remapped_labels = np.array([rank_map[l] for l in labels])
    return kmeans, scaler, remapped_labels, rank_map


def predict_cluster(kmeans, scaler, rank_map, age, smoke, drink):
    input_data = np.array([[age, smoke, drink]])
    input_scaled = scaler.transform(input_data)
    raw_label = kmeans.predict(input_scaled)[0]
    return rank_map[raw_label]


CLUSTER_INFO = {
    0: {"name": "매우 건강군", "color": "#22c55e", "box_class": "result-box",        "emoji": "✅"},
    1: {"name": "위험군",     "color": "#ef4444", "box_class": "result-box danger",  "emoji": "⚠️"},
    2: {"name": "건강군",     "color": "#3b82f6", "box_class": "result-box warning", "emoji": "🔵"},
}
COLORS = {0: "#eab308", 1: "#7c3aed", 2: "#0d9488"}  # 산점도용 색상


# ── 산점도 그리기 ────────────────────────────────────────────────
def draw_scatter(df, labels, patient_smoke, patient_drink):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for cluster_id, color in COLORS.items():
        mask = labels == cluster_id
        ax.scatter(
            df.loc[mask, "흡연량"],
            df.loc[mask, "음주량"],
            c=color, alpha=0.75, s=50, zorder=2,
        )

    # 현재 환자 위치 (별표)
    ax.scatter(
        patient_smoke, patient_drink,
        marker="*", s=280, c="#1d4ed8", zorder=5, label="현재 환자",
    )

    ax.set_xlabel("흡연량", fontsize=10)
    ax.set_ylabel("음주량(정규화)", fontsize=10)
    ax.set_title("군집 시각화", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    patches = [
        mpatches.Patch(color=COLORS[0], label="0번: 매우 건강군"),
        mpatches.Patch(color=COLORS[1], label="1번: 위험군"),
        mpatches.Patch(color=COLORS[2], label="2번: 건강군"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#1d4ed8",
                   markersize=12, label="현재 환자"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")

    plt.tight_layout()
    return fig


# ── 메인 UI ──────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫁 폐암 환자 군집 분석 시스템</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-desc">AI가 환자의 특성을 분석하여<br>어떤 군집(유형)에 속하는지 예측합니다.</div>',
    unsafe_allow_html=True,
)
st.divider()

# 데이터 & 모델 준비
df = generate_sample_data()
kmeans, scaler, labels, rank_map = train_model(df)

# ── 환자 정보 입력 ────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 환자 정보 입력</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("나이", min_value=0.0, max_value=120.0, value=50.0, step=1.0, format="%.2f")
with col2:
    smoke = st.number_input("흡연량", min_value=0.0, max_value=100.0, value=10.0, step=1.0, format="%.2f")
with col3:
    drink = st.number_input("음주량", min_value=0.0, max_value=50.0, value=5.0, step=1.0, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

# ── 분석 버튼 ─────────────────────────────────────────────────────
if st.button("🔍 군집 분석하기", use_container_width=True, type="primary"):
    cluster = predict_cluster(kmeans, scaler, rank_map, age, smoke, drink)
    info = CLUSTER_INFO[cluster]

    st.markdown(
        f'<div class="{info["box_class"]}">'
        f'{info["emoji"]} 이 환자는 <strong>{cluster}번 군집</strong>에 속합니다. ({info["name"]})'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="legend-text">0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.</span>',
        unsafe_allow_html=True,
    )

    # 산점도
    fig = draw_scatter(df, labels, smoke, drink)
    st.pyplot(fig)

    # 상세 설명
    with st.expander("📊 군집별 상세 설명 보기"):
        desc_data = {
            "군집": ["0번 — 매우 건강군", "1번 — 위험군", "2번 — 건강군"],
            "특징": [
                "낮은 흡연량·음주량, 비교적 젊은 연령대",
                "높은 흡연량, 중장년층, 폐암 위험 가장 높음",
                "보통 수준의 흡연량·음주량, 중간 위험도",
            ],
            "권고사항": [
                "현재 건강 습관 유지",
                "즉각적인 금연·절주 및 정밀 검진 권장",
                "정기 검진 및 생활습관 개선 권장",
            ],
        }
        st.table(pd.DataFrame(desc_data))
