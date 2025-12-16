import streamlit as st
import pandas as pd
import plotly.express as px

if st.button("⬅ Quay lại trang chính"):
    st.switch_page("app.py")

st.title("Tổng quan doanh thu & đơn hàng")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "D:/python_project/olist_app/data/master_data_final.csv"
    )
    # parse datetime
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df

df = load_data()

@st.cache_data
def load_geo():
    geo = pd.read_csv(
        "D:/python_project/olist_app/data/olist_geolocation_dataset.csv"
    )
    return geo

geo = load_geo()

# ================= FEATURE ENGINEERING =================
df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
df["order_value"] = df["price"] + df["freight_value"]

# ================= KPI =================
total_orders = df["order_id"].nunique()
total_revenue = df["order_value"].sum()
avg_order_value = df.groupby("order_id")["order_value"].sum().mean()
late_rate = df["is_late"].mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("🧾 Tổng đơn hàng", f"{total_orders:,}")
k2.metric("💰 Tổng doanh thu (BRL)", f"{total_revenue:,.2f}")
k3.metric("📦 Giá trị đơn TB", f"{avg_order_value:,.2f}")
k4.metric("⏱ Tỷ lệ giao trễ", f"{late_rate:.2f}%")

st.markdown("---")

# ================= ORDERS TREND =================
st.subheader("📈 Số đơn hàng theo thời gian")

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

# ================= REVENUE TREND =================
st.subheader("💰 Doanh thu theo thời gian")

revenue_trend = (
    df.groupby("month")["order_value"]
    .sum()
    .reset_index(name="total_revenue")
)

fig_revenue = px.line(
    revenue_trend,
    x="month",
    y="total_revenue",
    markers=True
)
st.plotly_chart(fig_revenue, use_container_width=True)

# ================= ORDER VALUE DISTRIBUTION =================
st.subheader("📦 Phân phối giá trị đơn hàng")

order_value_df = df.groupby("order_id")["order_value"].sum().reset_index()

# Chia nhóm giá trị đơn hàng
bins = [0, 50, 100, 200, 500, 1000, order_value_df["order_value"].max()]
labels = ["<50", "50–100", "100–200", "200–500", "500–1000", ">1000"]

order_value_df["value_group"] = pd.cut(
    order_value_df["order_value"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

value_dist = (
    order_value_df["value_group"]
    .value_counts()
    .sort_index()
    .reset_index()
)
value_dist.columns = ["value_group", "orders"]

fig_value_group = px.bar(
    value_dist,
    x="value_group",
    y="orders",
    title="Phân bố đơn hàng theo mức giá"
)

st.plotly_chart(fig_value_group, use_container_width=True)

# ================= DELIVERY ANALYSIS =================
st.subheader("🚚 Phân tích thời gian giao hàng")

d1, d2 = st.columns(2)

with d1:
    delivery_avg = (
        df.groupby("delivery_group")["delivery_days"]
        .mean()
        .reset_index()
    )

    fig_delivery_avg = px.bar(
        delivery_avg,
        x="delivery_group",
        y="delivery_days",
        title="Số ngày giao hàng trung bình theo trạng thái"
    )
    st.plotly_chart(fig_delivery_avg, use_container_width=True)

with d2:
    late_count = df["delivery_group"].value_counts().reset_index()
    late_count.columns = ["delivery_group", "count"]

    fig_late = px.pie(
        late_count,
        names="delivery_group",
        values="count",
        hole=0.4,
        title="Tỷ lệ giao hàng đúng hạn / trễ"
    )
    st.plotly_chart(fig_late, use_container_width=True)

# ================= REVIEW ANALYSIS =================
st.subheader("⭐ Đánh giá đơn hàng")

r1, r2 = st.columns(2)

with r1:
    review_score = (
        df.groupby("review_score")["order_id"]
        .nunique()
        .reset_index(name="orders")
    )

    fig_review = px.bar(
        review_score,
        x="review_score",
        y="orders",
        title="Số đơn theo điểm đánh giá"
    )
    st.plotly_chart(fig_review, use_container_width=True)

with r2:
    review_group = (
        df.groupby("review_group")["order_id"]
        .nunique()
        .reset_index(name="orders")
    )

    fig_review_group = px.pie(
        review_group,
        names="review_group",
        values="orders",
        hole=0.4,
        title="Nhóm đánh giá"
    )
    st.plotly_chart(fig_review_group, use_container_width=True)

# ================= TOP CATEGORY =================
st.subheader("🏆 Top 10 danh mục sản phẩm theo doanh thu")

top_category = (
    df.groupby("product_category_name")["order_value"]
    .sum()
    .reset_index()
    .sort_values(by="order_value", ascending=False)
    .head(10)
)

fig_cat = px.bar(
    top_category,
    x="order_value",
    y="product_category_name",
    orientation="h"
)
st.plotly_chart(fig_cat, use_container_width=True)

# Đếm số đơn theo bang
orders_by_state = (
    df.groupby("customer_state")["order_id"]
    .nunique()
    .reset_index(name="total_orders")
)

# Lấy lat/lng trung bình theo bang
geo_state = (
    geo.groupby("geolocation_state")[["geolocation_lat", "geolocation_lng"]]
    .mean()
    .reset_index()
)

# Merge
map_df = orders_by_state.merge(
    geo_state,
    left_on="customer_state",
    right_on="geolocation_state",
    how="left"
)

st.subheader("🗺️ Phân bố đơn hàng theo khu vực")

fig_map = px.scatter_mapbox(
    map_df,
    lat="geolocation_lat",
    lon="geolocation_lng",
    size="total_orders",
    color="total_orders",
    hover_name="customer_state",
    hover_data={"total_orders": True},
    zoom=3,
    height=550
)

fig_map.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":0,"l":0,"b":0}
)

st.plotly_chart(fig_map, use_container_width=True)

# ===== DOANH THU THEO BANG =====
revenue_by_state = (
    df.groupby("customer_state")["order_value"]
    .sum()
    .reset_index(name="total_revenue")
)

# Lấy lat/lng trung bình theo bang
geo_state = (
    geo.groupby("geolocation_state")[["geolocation_lat", "geolocation_lng"]]
    .mean()
    .reset_index()
)

# Merge doanh thu + tọa độ
map_revenue_df = revenue_by_state.merge(
    geo_state,
    left_on="customer_state",
    right_on="geolocation_state",
    how="left"
)

st.subheader("🗺️ Phân bố doanh thu theo bang")

fig_map_revenue = px.scatter_mapbox(
    map_revenue_df,
    lat="geolocation_lat",
    lon="geolocation_lng",
    size="total_revenue",
    color="total_revenue",
    hover_name="customer_state",
    hover_data={"total_revenue": ":,.0f"},
    zoom=3,
    height=550
)

fig_map_revenue.update_layout(
    mapbox_style="carto-positron",
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig_map_revenue, use_container_width=True)
