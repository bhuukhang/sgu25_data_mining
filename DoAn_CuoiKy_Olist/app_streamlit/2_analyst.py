import streamlit as st
import pandas as pd
import plotly.express as px

if st.button("⬅ Quay lại trang chính"):
    st.switch_page("app.py")

st.title("📈 Phân tích hiệu quả kinh doanh (Business Analysis)")

st.markdown("""
**phục vụ ra quyết định kinh doanh**
""")

# ================================
# Load dữ liệu file FULL 37 cột
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "D:/python_project/olist_app/data/master_data_final.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date"
        ]
    )
    return df

df = load_data()

# Feature
df["order_value"] = df["price"] + df["freight_value"]
df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

# ===============================
# 1. KPI TỔNG QUAN
# ===============================
st.subheader("📌 Tổng quan hiệu quả kinh doanh")

total_orders = df["order_id"].nunique()
total_revenue = df["order_value"].sum()
avg_order_value = df.groupby("order_id")["order_value"].sum().mean()
late_rate = df["is_late"].mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("🧾 Tổng số đơn", f"{total_orders:,}")
k2.metric("💰 Tổng doanh thu", f"{total_revenue:,.0f} BRL")
k3.metric("📦 Giá trị đơn TB", f"{avg_order_value:,.0f} BRL")
k4.metric("⏱ Tỷ lệ giao trễ", f"{late_rate:.2f}%")

st.info("""
📌 **Insight nhanh**  
- Doanh thu tập trung chủ yếu vào một số danh mục và khu vực lớn  
- Tỷ lệ giao trễ vẫn còn đáng kể → ảnh hưởng trải nghiệm khách hàng
""")

st.markdown("---")

# ===============================
# 2. XU HƯỚNG ĐƠN HÀNG
# ===============================
st.subheader("📈 Xu hướng số đơn hàng theo thời gian")

orders_trend = (
    df.groupby("month")["order_id"]
    .nunique()
    .reset_index(name="total_orders")
)

fig_orders = px.line(
    orders_trend,
    x="month",
    y="total_orders",
    markers=True
)
st.plotly_chart(fig_orders, use_container_width=True)

# ===============================
# 3. XU HƯỚNG DOANH THU
# ===============================
st.subheader("💰 Xu hướng doanh thu theo thời gian")

revenue_trend = (
    df.groupby("month")["order_value"]
    .sum()
    .reset_index(name="revenue")
)

fig_revenue = px.line(
    revenue_trend,
    x="month",
    y="revenue",
    markers=True
)
st.plotly_chart(fig_revenue, use_container_width=True)

st.markdown("---")

# ===============================
# 4. DANH MỤC TẠO DOANH THU
# ===============================
st.subheader("🏆 Danh mục đóng góp doanh thu lớn nhất")

top_category = (
    df.groupby("product_category_name")["order_value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_cat = px.bar(
    top_category,
    x="order_value",
    y="product_category_name",
    orientation="h",
    labels={"order_value": "Doanh thu (BRL)", "product_category_name": "Danh mục"}
)
st.plotly_chart(fig_cat, use_container_width=True)

st.info("""
📌 **Insight**  
Một số danh mục không có nhiều đơn nhưng mang lại doanh thu cao →  
phù hợp chiến lược **tối ưu giá trị đơn hàng**.
""")

st.markdown("---")

# ===============================
# 5. GIAO HÀNG & TRẢI NGHIỆM KHÁCH HÀNG
# ===============================
st.subheader("🚚 Hiệu suất giao hàng & tác động")

c1, c2 = st.columns(2)

with c1:
    delivery_avg = (
        df.groupby("delivery_group")["delivery_days"]
        .mean()
        .reset_index()
    )

    fig_delivery = px.bar(
        delivery_avg,
        x="delivery_group",
        y="delivery_days",
        title="Thời gian giao hàng trung bình"
    )
    st.plotly_chart(fig_delivery, use_container_width=True)

with c2:
    review_by_delivery = (
        df.groupby("delivery_group")["review_score"]
        .mean()
        .reset_index()
    )

    fig_review = px.bar(
        review_by_delivery,
        x="delivery_group",
        y="review_score",
        title="Điểm đánh giá trung bình theo trạng thái giao hàng"
    )
    st.plotly_chart(fig_review, use_container_width=True)

st.info("""
📌 **Insight quan trọng**  
Giao hàng trễ có mối liên hệ rõ ràng với điểm đánh giá thấp →  
cải thiện logistics giúp tăng sự hài lòng khách hàng.
""")

st.markdown("---")

# ===============================
# 6. KẾT LUẬN KINH DOANH
# ===============================
st.subheader("🧠 Kết luận & gợi ý chiến lược")

st.success("""
✔ Tập trung vào danh mục mang lại doanh thu cao  
✔ Tối ưu giao hàng để cải thiện trải nghiệm & đánh giá  
✔ Ưu tiên mở rộng tại các khu vực có giá trị kinh tế lớn  
""")