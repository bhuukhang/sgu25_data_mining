import streamlit as st

st.set_page_config(
    page_title="Olist Dashboard",
    page_icon="📊",
    layout="wide"
)

# ===== HEADER =====
st.markdown(
    """
    <h1 style="text-align:center;">📊 Olist Dashboard</h1>
    <p style="text-align:center; color:gray; font-size:16px;">
        Hệ thống phân tích dữ liệu bán hàng Olist
    </p>
    <br>
    """,
    unsafe_allow_html=True
)

# ===== ICON MENU =====
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        <div style="text-align:center; font-size:80px;">💰</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h4 style='text-align:center;'>Doanh thu</h4>", unsafe_allow_html=True)
    if st.button("Xem chi tiết", key="rev"):
        st.switch_page("pages/1_overview.py")
with col2:
    st.markdown(
        """
        <div style="text-align:center; font-size:80px;">👥</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h4 style='text-align:center;'>Doanh nghiệp</h4>", unsafe_allow_html=True)
    if st.button("Xem chi tiết", key="cus"):
        st.switch_page("pages/2_analyst.py")
with col3:
    st.markdown(
        """
        <div style="text-align:center; font-size:80px;">📊</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h4 style='text-align:center;'>Phân loại khách hàng</h4>", unsafe_allow_html=True)
    if st.button("Xem chi tiết", key="rfm"):
        st.switch_page("pages/3_customer_rfm.py")
with col4:
    st.markdown(
        """
        <div style="text-align:center; font-size:80px;">🔗</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h4 style='text-align:center;'>Sản phẩm</h4>", unsafe_allow_html=True)
    if st.button("Xem chi tiết", key="apr"):
        st.switch_page("pages/4_apriori.py")
st.markdown("---")
with col5:
    st.markdown(
        """
        <div style="text-align:center; font-size:80px;">😊</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h4 style='text-align:center;'>Sự hài lòng</h4>", unsafe_allow_html=True)
    if st.button("Xem chi tiết", key="satis"):
        st.switch_page("pages/5_customer_satisfaction.py")
st.markdown("---")