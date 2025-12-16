import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



# ============================
# NAV
# ============================
if st.button("⬅ Quay lại trang chính"):
    st.switch_page("app.py")

st.title("😊 Phân loại sự hài lòng của khách hàng")

st.markdown(
    """
    Trang này sử dụng **Random Forest đã huấn luyện**  
    để **dự đoán mức độ hài lòng của khách hàng** dựa trên dữ liệu đơn hàng.
    """
)

# ============================
# LOAD MODEL + IMPUTER
# ============================
@st.cache_resource
def load_models():
    model = joblib.load("D:/python_project/olist_app/pages/best_rf_model.pkl")
    imputer = joblib.load("D:/python_project/olist_app/pages/imputer_master.joblib")
    return model, imputer

model, imputer = load_models()

# ============================
# LOAD DATA
# ============================
DATA_PATH = "D:/python_project/olist_app/data/master_data.csv"
df = pd.read_csv(DATA_PATH)

# Tạo feature product_volume_cm3
df['product_volume_cm3'] = (
    df['product_length_cm'].fillna(0) *
    df['product_height_cm'].fillna(0) *
    df['product_width_cm'].fillna(0)
)

potential_features = [
    'price', 'freight_value', 'delivery_days', 'is_late',
    'product_weight_g', 'product_volume_cm3',
    'payment_value_sum', 'payment_count'
]

features = [c for c in potential_features if c in df.columns]

df = df.dropna(subset=features)

X = df[features]
X_imp = pd.DataFrame(
    imputer.transform(X),
    columns=features,
    index=X.index
)

# ============================
# DỰ ĐOÁN
# ============================
df["predicted_label"] = model.predict(X_imp)
df["predicted_proba"] = model.predict_proba(X_imp)[:, 1]

label_map = {0: "😡 Không hài lòng", 1: "😊 Hài lòng"}
df["prediction"] = df["predicted_label"].map(label_map)

# ============================
# KPI OVERVIEW
# ============================
st.subheader("📊 Tổng quan kết quả dự đoán")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "😊 Tỷ lệ hài lòng",
        f"{(df['predicted_label'].mean()*100):.1f}%"
    )

with col2:
    st.metric(
        "😡 Không hài lòng",
        f"{(1 - df['predicted_label'].mean())*100:.1f}%"
    )

with col3:
    st.metric(
        "📦 Số đơn hàng",
        f"{len(df):,}"
    )

# ============================
# BIỂU ĐỒ PHÂN PHỐI
# ============================
st.subheader("📈 Phân phối mức độ hài lòng")

fig, ax = plt.subplots(figsize=(5,4))
sns.countplot(
    data=df,
    x="prediction",
    palette={"😡 Không hài lòng":"#ff6b6b", "😊 Hài lòng":"#51cf66"},
    ax=ax
)
ax.set_xlabel("")
ax.set_ylabel("Số lượng đơn hàng")
st.pyplot(fig)

# ============================
# FEATURE IMPORTANCE
# ============================
st.subheader("🔍 Yếu tố ảnh hưởng đến sự hài lòng")

importances = model.feature_importances_
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(7,4))
sns.barplot(
    data=imp_df,
    x="Importance",
    y="Feature",
    palette="Blues_r",
    ax=ax2
)
st.pyplot(fig2)

# ============================
# XEM CHI TIẾT DỰ ĐOÁN
# ============================
st.subheader("🧾 Xem kết quả dự đoán chi tiết")

st.dataframe(
    df[features + ["prediction", "predicted_proba"]]
    .sample(20)
    .reset_index(drop=True)
)