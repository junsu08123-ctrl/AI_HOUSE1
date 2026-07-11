import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import folium
from streamlit_folium import st_folium


# -----------------------------
# 한글 폰트 설정
# -----------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="서울 AI 부동산 예측",
    page_icon="🏢",
    layout="wide"
)


# -----------------------------
# 모델 불러오기
# -----------------------------
model = joblib.load("model.pkl")


# -----------------------------
# 결과 저장 공간
# -----------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "age" not in st.session_state:
    st.session_state.age = 10

if "area" not in st.session_state:
    st.session_state.area = 84.0

if "floor" not in st.session_state:
    st.session_state.floor = 10



# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.title("🏢 AI Real Estate")

st.sidebar.info(
"""
서울시 건축물대장과
실거래가 데이터를 학습한

Random Forest 기반
부동산 가격 예측 시스템
"""
)



# -----------------------------
# 제목
# -----------------------------
st.title("🏢 서울 AI 부동산 가격 예측 시스템")

st.write(
"""
건축물 정보를 입력하면
AI가 예상 거래가격을 예측합니다.
"""
)


st.divider()



# -----------------------------
# 입력
# -----------------------------
st.subheader("🏠 건축물 정보 입력")


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "🏗️ 건물 나이(년)",
        min_value=0,
        max_value=100,
        value=st.session_state.age
    )


with col2:

    area = st.number_input(
        "📐 전용면적(㎡)",
        min_value=10.0,
        max_value=300.0,
        value=st.session_state.area
    )


with col3:

    floor = st.number_input(
        "🏢 층",
        min_value=1,
        max_value=80,
        value=st.session_state.floor
    )

col4, col5, col6 = st.columns(3)


with col4:

    households = st.number_input(
        "🏠 세대수",
        min_value=1,
        value=100
    )


with col5:

    total_area = st.number_input(
        "📐 연면적(㎡)",
        min_value=10.0,
        value=5000.0
    )


with col6:

    ground_floor = st.number_input(
        "🏢 지상층수",
        min_value=1,
        value=10
    )

st.divider()



# -----------------------------
# 예측 버튼
# -----------------------------
if st.button(
    "🔮 AI 가격 예측",
    use_container_width=True
):

    input_data = pd.DataFrame(
    {
        "건물나이":[age],
        "전용면적(㎡)":[area],
        "층":[floor],
        "세대수":[households],
        "연면적":[total_area],
        "지상층수":[ground_floor]
    }
)

    result = model.predict(input_data)[0]


    st.session_state.prediction = result
    st.session_state.age = age
    st.session_state.area = area
    st.session_state.floor = floor



# -----------------------------
# 여기부터 결과 출력
# -----------------------------
if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    age = st.session_state.age
    area = st.session_state.area
    floor = st.session_state.floor
        # -----------------------------
    # 가격 결과
    # -----------------------------

    st.subheader("💰 AI 예측 결과")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "예상 거래가격",
            f"{prediction:,.2f} 억원"
        )


    with col2:

        st.metric(
            "약 환산 가격",
            f"{prediction*10000:,.0f} 만원"
        )



    # -----------------------------
    # 가격 범위
    # -----------------------------

    lower_price = prediction * 0.9
    upper_price = prediction * 1.1


    st.info(
f"""
📊 AI 예상 가격 범위

{lower_price:,.2f}억원 ~ {upper_price:,.2f}억원

(예측값 기준 ±10%)
"""
    )


    st.success(
        "✅ AI 예측 완료"
    )


    st.divider()



    # -----------------------------
    # 입력 정보
    # -----------------------------

    st.subheader("📋 분석한 건축물 정보")


    info = pd.DataFrame(
        {
            "항목":[
                "건물 나이",
                "전용면적",
                "층"
            ],

            "입력값":[
                f"{age}년",
                f"{area}㎡",
                f"{floor}층"
            ]
        }
    )


    st.table(info)



    # -----------------------------
    # AI 분석
    # -----------------------------

    st.subheader("🤖 AI 분석")


    if age >= 30:

        age_text = (
            "건축 연령이 높아 "
            "노후 영향이 가격에 반영될 가능성이 있습니다."
        )

    else:

        age_text = (
            "비교적 최근 건축물로 "
            "노후 영향은 낮습니다."
        )



    if floor >= 10:

        floor_text = (
            "높은 층수로 인해 "
            "선호도가 높을 가능성이 있습니다."
        )

    else:

        floor_text = (
            "저층 건물 특성을 가지고 있습니다."
        )



    st.write(
f"""
- {age_text}

- 전용면적 {area}㎡ 기준으로 분석되었습니다.

- {floor_text}
"""
    )



    # -----------------------------
    # 변수 중요도
    # -----------------------------

    if hasattr(model, "feature_importances_"):


        st.subheader(
            "📊 가격 영향 변수 분석"
        )


        importance = pd.DataFrame(
            {
                "변수": model.feature_names_in_,
                "중요도": model.feature_importances_
            }
        )


        importance = importance.sort_values(
            "중요도",
            ascending=True
        )


        fig, ax = plt.subplots()


        ax.barh(
            importance["변수"],
            importance["중요도"]
        )


        ax.set_title(
            "AI가 판단한 가격 영향도"
        )


        ax.set_xlabel(
            "Importance"
        )


        st.pyplot(fig)



    # -----------------------------
    # 면적 변화 그래프
    # -----------------------------

    st.subheader(
        "📈 전용면적 변화에 따른 가격 변화"
    )


    area_range = np.linspace(
        max(10, area-30),
        area+50,
        20
    )


    prices = []

for a in area_range:

    test = pd.DataFrame(
        {
            "건물나이":[age],
            "전용면적(㎡)":[a],
            "층":[floor],
            "세대수":[households],
            "연면적":[total_area],
            "지상층수":[ground_floor]
        }
    )

    p = model.predict(test)[0]

    prices.append(p)

        p = model.predict(test)[0]


        prices.append(p)



    fig2, ax2 = plt.subplots()


    ax2.plot(
        area_range,
        prices,
        marker="o"
    )


    ax2.set_xlabel(
        "전용면적(㎡)"
    )


    ax2.set_ylabel(
        "예상 가격(억원)"
    )


    ax2.set_title(
        "면적 증가에 따른 가격 변화"
    )


    st.pyplot(fig2)
    # -----------------------------
    # 서울 지도 표시
    # -----------------------------

    st.subheader(
        "🗺️ AI 부동산 예측 지도"
    )


    district = st.selectbox(
        "📍 자치구 선택",
        [
            "강남구",
            "서초구",
            "송파구",
            "마포구",
            "용산구",
            "영등포구",
            "성동구",
            "노원구",
            "강서구"
        ]
    )


    district_location = {

        "강남구":[37.5172,127.0473],
        "서초구":[37.4837,127.0324],
        "송파구":[37.5145,127.1059],
        "마포구":[37.5663,126.9014],
        "용산구":[37.5326,126.9905],
        "영등포구":[37.5264,126.8963],
        "성동구":[37.5633,127.0367],
        "노원구":[37.6542,127.0568],
        "강서구":[37.5509,126.8495]

    }


    map_location = district_location[district]


    m = folium.Map(
        location=map_location,
        zoom_start=13
    )


    folium.CircleMarker(

        location=map_location,

        radius=max(10, prediction*3),

        popup=f"""
        🏢 AI 예측 건물<br><br>
        📍 위치 : {district}<br>
        💰 예상 가격 : {prediction:.2f}억원<br>
        📐 면적 : {area}㎡<br>
        🏢 층수 : {floor}층<br>
        🏗️ 건물 나이 : {age}년
        """,

        tooltip="AI 예측 위치",

        fill=True

    ).add_to(m)


    st_folium(
        m,
        width=900,
        height=600
    )
