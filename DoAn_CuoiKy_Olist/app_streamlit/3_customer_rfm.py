import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

if st.button("⬅ Quay lại trang chính"):
    st.switch_page("app.py")

st.title("📊 Phân loại khách hàng")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv(
        "D:/python_project/olist_app/data/master_data_final.csv",
        parse_dates=["order_purchase_timestamp"]
    )

@st.cache_resource
def load_model():
    scaler = joblib.load("D:/python_project/olist_app/pages/rfm_scaler_final.pkl")
    kmeans = joblib.load("D:/python_project/olist_app/pages/rfm_kmeans_final.pkl")
    return scaler, kmeans

df = load_data()
scaler, kmeans = load_model()

# ================= RFM =================
df["total_value"] = df["price"] + df["freight_value"]
snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_unique_id").agg({
    "order_purchase_timestamp": lambda x: (snapshot_date - x.max()).days,
    "order_id": "nunique",
    "total_value": "sum"
}).reset_index()

rfm.columns = ["customer_unique_id", "Recency", "Frequency", "Monetary"]
rfm = rfm[rfm["Monetary"] > 0]

# ================= PREDICT =================
rfm_log = np.log(rfm[["Recency", "Frequency", "Monetary"]] + 1)
rfm_scaled = scaler.transform(rfm_log)

rfm["Cluster"] = kmeans.predict(rfm_scaled)

segment_map = {
    0: "Khách hàng vãng lai",
    1: "Khách hàng tiềm năng",
    2: "Khách hàng trung thành",
    3: "Khách hàng VIP"
}
rfm["Segment"] = rfm["Cluster"].map(segment_map)

segment_order = [
    "Khách hàng vãng lai",
    "Khách hàng tiềm năng",
    "Khách hàng trung thành",
    "Khách hàng VIP"
]

rfm["Segment"] = pd.Categorical(
    rfm["Segment"],
    categories=segment_order,
    ordered=True
)

# ================= KPI =================
c1, c2, c3 = st.columns(3)
c1.metric("Tổng khách hàng", f"{rfm.shape[0]:,}")
c2.metric("Số phân khúc", rfm["Segment"].nunique())
c3.metric("Khách VIP", (rfm["Segment"] == "Khách hàng VIP").sum())

st.divider()

# ================= BIỂU ĐỒ =================
col1, col2 = st.columns(2)

with col1:
    fig_count = px.pie(
        rfm,
        names="Segment",
        title="Tỷ lệ khách hàng theo phân khúc",
        hole=0.4
    )
    st.plotly_chart(fig_count, use_container_width=True)

with col2:
    seg_value = (
    rfm.groupby("Segment", observed=True)["Monetary"]
       .mean()
       .reset_index()
    )
    fig_money = px.bar(
        seg_value,
        x="Segment",
        y="Monetary",
        title="Chi tiêu trung bình theo phân khúc"
    )
    st.plotly_chart(fig_money, use_container_width=True)

st.divider()

# ================= CLUSTER VISUAL =================
st.subheader("📌 Biểu đồ phân cụm khách hàng")

tab1, tab2 = st.tabs(["2D (Recency vs Monetary)", "3D RFM"])

with tab1:
    fig_2d = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color="Segment",
        opacity=0.6,
        title="Phân cụm khách hàng (Recency – Monetary)",
        labels={
            "Recency": "Recency (ngày)",
            "Monetary": "Tổng chi tiêu"
        }
    )
    fig_2d.update_layout(yaxis_tickformat=",.0f")
    st.plotly_chart(fig_2d, use_container_width=True)

with tab2:
    fig_3d = px.scatter_3d(
        rfm,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Segment",
        opacity=0.7,
        title="Phân cụm khách hàng 3D (RFM)"
    )
    fig_3d.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# ================= SEGMENT INSIGHT (REPLACE CUSTOMER LIST) =================
st.subheader("📊 Phân tích chi tiết phân khúc khách hàng")

selected_segment = st.selectbox(
    "Chọn phân khúc khách hàng:",
    segment_order
)

seg_df = rfm[rfm["Segment"] == selected_segment]

# --- KPI mini ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Số khách", f"{seg_df.shape[0]:,}")
k2.metric("⏱ Recency TB", f"{seg_df['Recency'].mean():.0f} ngày")
k3.metric("🔁 Frequency TB", f"{seg_df['Frequency'].mean():.1f}")
k4.metric("💰 Monetary TB", f"{seg_df['Monetary'].mean():,.0f}")

st.divider()

# ---------- BIỂU ĐỒ HÀNH VI ----------
col1, col2 = st.columns(2)

with col1:
    fig_dist = px.histogram(
        seg_df,
        x="Monetary",
        nbins=40,
        title="Phân bố chi tiêu khách hàng"
    )
    fig_dist.update_layout(xaxis_tickformat=",.0f")
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("📊 Hành vi trung bình của phân khúc")

    rfm_simple = pd.DataFrame({
        "Chỉ số": ["Recency (ngày)", "Frequency (lần)", "Monetary (BRL)"],
        "Giá trị trung bình": [
            seg_df["Recency"].mean(),
            seg_df["Frequency"].mean(),
            seg_df["Monetary"].mean()
        ]
    })

    fig_simple = px.bar(
        rfm_simple,
        x="Chỉ số",
        y="Giá trị trung bình",
        text_auto=".2s",
        title="3 chỉ số RFM của phân khúc"
    )

    fig_simple.update_layout(yaxis_tickformat=",.0f")
    st.plotly_chart(fig_simple, use_container_width=True)

# ---------- TÓM TẮT HÀNH VI PHÂN KHÚC ----------
st.subheader("📌 Tóm tắt hành vi phân khúc")

c1, c2, c3 = st.columns(3)

c1.metric(
    "🕒 Lần mua gần nhất (TB)",
    f"{seg_df['Recency'].mean():.0f} ngày"
)

c2.metric(
    "🔁 Số lần mua (TB)",
    f"{seg_df['Frequency'].mean():.1f} lần"
)

c3.metric(
    "💰 Chi tiêu (TB)",
    f"{seg_df['Monetary'].mean():,.0f}"
)

st.caption(
    "Ba chỉ số trên cho biết mức độ quay lại, tần suất mua và giá trị chi tiêu trung bình của phân khúc."
)

# ---------- INSIGHT TỰ ĐỘNG ----------
st.subheader("🧠 Nhận định & gợi ý")

if selected_segment == "Khách hàng vãng lai":
    st.warning(
        "Khách hàng ít mua và đã lâu chưa quay lại. "
        "Nên dùng mã giảm giá hoặc email nhắc mua để kích hoạt lại."
    )

elif selected_segment == "Khách hàng tiềm năng":
    st.info(
        "Khách hàng có tần suất mua khá. "
        "Nếu đẩy upsell hoặc combo phù hợp, có thể chuyển thành khách trung thành."
    )

elif selected_segment == "Khách hàng trung thành":
    st.success(
        "Khách hàng mua đều và ổn định. "
        "Nên duy trì ưu đãi định kỳ để giữ chân."
    )

elif selected_segment == "Khách hàng VIP":
    st.success(
        "Nhóm khách có giá trị cao nhất. "
        "Cần chương trình VIP riêng, quà tặng và chăm sóc đặc biệt."
    )