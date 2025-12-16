import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

if st.button("⬅ Quay lại trang chính"):
    st.switch_page("app.py")

st.title("📦 Top sản phẩm bán chạy")

# ============================
# 1. Đọc file gốc tự động
# ============================

FILE_PATH = "D:/python_project/olist_app/data/master_data_final.csv"

try:
    df = pd.read_csv(FILE_PATH)
except:
    st.error("❌ Không thể đọc file gốc. Kiểm tra lại đường dẫn FILE_PATH!")
    st.stop()


# ============================
# 2. Gom list sản phẩm theo order_id
# ============================


df_grouped = df.groupby("order_id")["product_category_name"].apply(list).reset_index()

# ============================
# 3. Load mô hình Apriori đã train (PKL)
# ============================

try:
    with open("D:/python_project/olist_app/pages/top_combo.pkl", "rb") as f:
        top_combo = pickle.load(f)

    with open("D:/python_project/olist_app/pages/top_rules.pkl", "rb") as f:
        top_rules = pickle.load(f)
    # 🔧 FIX lỗi: frozenset không convert được sang pyarrow
    for col in ["antecedents", "consequents"]:
        if col in top_rules.columns:
            top_rules[col] = top_rules[col].apply(lambda x: list(x) if isinstance(x, (set, frozenset)) else x)

except:
    st.error("❌ Không tìm thấy file top_combo.pkl hoặc top_rules.pkl trong thư mục ứng dụng.")
    st.stop()

# ============================
# 4. Dashboard combo bán chạy
# ============================

st.subheader("🔥 Top Combo Sản Phẩm Bán Chạy")
st.dataframe(
    top_combo[["itemsets_str", "support", "length"]].rename(
    columns={"itemsets_str": "Combo Sản Phẩm", "support": "Tỷ lệ xuất hiện", "length": "Số lượng sản phẩm"}))
# ============================
# 7. Gợi ý sản phẩm phổ biến (AUTO)
# ============================

st.subheader("🛒 Sản phẩm thường được mua kèm")

st.markdown(
    """
    Danh sách gợi ý dưới đây được tổng hợp từ **các combo mua kèm phổ biến nhất**
    trong dữ liệu bán hàng.
    """
)

# Lấy top luật mạnh nhất
recommend_rules = (
    top_rules[top_rules["lift"] > 1]
    .sort_values(["confidence", "lift"], ascending=False)
    .head(12)
)

if recommend_rules.empty:
    st.info("Không có dữ liệu gợi ý.")
else:
    cols = st.columns(2)

    for i, (_, row) in enumerate(recommend_rules.iterrows()):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div style="
                    padding: 14px;
                    margin-bottom: 12px;
                    border-radius: 12px;
                    background-color: #f9fafb;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                ">
                    <div style="font-size:15px;">
                        <b>🛍 Mua:</b> {row['antecedents_str']}
                    </div>
                    <div style="font-size:16px; color:#ff4b4b; margin-top:4px;">
                        <b>👉 Gợi ý:</b> {row['consequents_str']}
                    </div>
                    <div style="font-size:12px; color:gray; margin-top:6px;">
                        Confidence: {row['confidence']:.2f} | Lift: {row['lift']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )