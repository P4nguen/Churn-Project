import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap
from catboost import CatBoostClassifier

# -----------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE STİL (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Müşteri Sadakat Analizi (Churn AI)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI için Özel CSS
st.markdown("""
    <style>
    /* Genel Arka Plan ve Font */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar Stili */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    [data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
    
    /* Özel Kart Tasarımı */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #3498db;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Tab Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px;
        color: #2c3e50;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR VE MODEL YÜKLEME
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    """Model ve encoder dosyalarını yükler."""
    try:
        with open('churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('categorical_cols.pkl', 'rb') as f:
            cat_cols = pickle.load(f)
        return model, encoders, features, cat_cols
    except FileNotFoundError:
        st.error("🚨 HATA: Model dosyaları (pkl) bulunamadı. Lütfen dizini kontrol edin.")
        return None, None, None, None

def create_gauge(probability):
    """Modern bir hız göstergesi grafiği oluşturur."""
    color = "#2ecc71" if probability < 0.3 else "#f1c40f" if probability < 0.7 else "#e74c3c"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Kayıp Riski (Churn)", 'font': {'size': 20, 'color': "#2c3e50"}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ecf0f1",
            'steps': [
                {'range': [0, 30], 'color': '#ebfbf2'},
                {'range': [30, 70], 'color': '#fef9e7'},
                {'range': [70, 100], 'color': '#fdedeb'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def get_smart_recommendations(prob, inputs):
    """Veriye dayalı akıllı öneriler sunar."""
    recs = []
    
    # Genel Risk Önerileri
    if prob > 0.7:
        recs.append(("🔴 KRİTİK SEVİYE", "Müşteri yüksek risk grubunda. Acil elde tutma (retention) ekibi aranmalı."))
        recs.append(("💸 İndirim", "Bir sonraki faturada %25 indirim teklif edin."))
    elif prob > 0.4:
        recs.append(("🟡 DİKKAT", "Müşteri risk belirtileri gösteriyor."))
        recs.append(("🎁 Kampanya", "Mevcut paketine ek 1 ay ücretsiz premium özellik sunun."))
    else:
        recs.append(("🟢 GÜVENLİ", "Müşteri sadakati yüksek görünüyor."))
        recs.append(("⭐ Referans", "Memnuniyet anketi gönderip arkadaşını getirmesini isteyin."))

    # Özellik Bazlı Spesifik Öneriler
    if inputs['MonthlyCharges'] > 90:
        recs.append(("💰 Fiyat Hassasiyeti", "Aylık ödemesi yüksek. Daha uygun fiyatlı uzun vadeli bir paket önerin."))
    
    if inputs['Contract'] == "Month-to-month":
        recs.append(("📝 Sözleşme", "Aylık sözleşme riski artırıyor. 1 veya 2 yıllık taahhüt için avantaj sunun."))
        
    if inputs['InternetService'] == "Fiber optic" and inputs['TechSupport'] == "No":
        recs.append(("🔧 Teknik Destek", "Fiber kullanıcısı ama teknik desteği yok. Destek paketi satmayı deneyin."))

    return recs

# -----------------------------------------------------------------------------
# 3. ANA UYGULAMA
# -----------------------------------------------------------------------------
def main():
    model, label_encoders, feature_names, categorical_cols = load_model_assets()
    if not model: return

    # --- Header Bölümü ---
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("🤖 AI Müşteri Analiz Paneli")
        st.markdown("*Yapay Zeka Destekli Churn (Müşteri Kaybı) Tahmin Sistemi*")
    with col_h2:
        st.markdown("") # Logo alanı olarak kullanılabilir

    st.markdown("---")

    # --- Sidebar: Veri Girişi ---
    st.sidebar.header("🛠️ Müşteri Profili Oluştur")
    
    # Giriş verilerini tutacak sözlük
    input_data = {}

    with st.sidebar.expander("👤 Kişisel Bilgiler", expanded=True):
        # UI'da Türkçe göster, Model için İngilizce kaydet
        gender_ui = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        input_data['gender'] = "Female" if gender_ui == "Kadın" else "Male"
        
        senior_ui = st.toggle("65 Yaş Üstü (Senior)")
        input_data['SeniorCitizen'] = 1 if senior_ui else 0
        
        partner_ui = st.selectbox("Medeni Durum / Partner", ["Yok", "Var"])
        input_data['Partner'] = "Yes" if partner_ui == "Var" else "No"
        
        dep_ui = st.selectbox("Bakmakla Yükümlü Olduğu Kişi", ["Yok", "Var"])
        input_data['Dependents'] = "Yes" if dep_ui == "Var" else "No"

    with st.sidebar.expander("📡 Hizmet Detayları"):
        tenure = st.slider("Abonelik Süresi (Ay)", 0, 72, 12, help="Müşterinin kaç aydır hizmet aldığı")
        input_data['tenure'] = tenure
        
        phone_ui = st.selectbox("Telefon Hizmeti", ["Yok", "Var"])
        input_data['PhoneService'] = "Yes" if phone_ui == "Var" else "No"
        
        # İnternet Servisi
        internet_map = {"Yok": "No", "DSL": "DSL", "Fiber Optik": "Fiber optic"}
        internet_ui = st.selectbox("İnternet Altyapısı", list(internet_map.keys()))
        input_data['InternetService'] = internet_map[internet_ui]
        
        # Ek Hizmetler (Çoklu seçim mantığı yerine tek tek soruyoruz model gereği)
        input_data['OnlineSecurity'] = "Yes" if st.checkbox("Online Güvenlik") else "No"
        input_data['TechSupport'] = "Yes" if st.checkbox("Teknik Destek") else "No"
        input_data['StreamingTV'] = "Yes" if st.checkbox("TV Yayını") else "No"
        
        # Diğer zorunlu alanlar için varsayılanlar (Sadelik için gizlendi veya basitleştirildi)
        # Modelin beklediği ama UI'da kalabalık etmemesi için arkada doldurulanlar:
        input_data['MultipleLines'] = "No" # Varsayılan
        input_data['OnlineBackup'] = "No"
        input_data['DeviceProtection'] = "No"
        input_data['StreamingMovies'] = "No"

    with st.sidebar.expander("💳 Ödeme & Sözleşme", expanded=True):
        contract_map = {"Aylık": "Month-to-month", "1 Yıllık": "One year", "2 Yıllık": "Two year"}
        contract_ui = st.selectbox("Sözleşme Tipi", list(contract_map.keys()))
        input_data['Contract'] = contract_map[contract_ui]
        
        input_data['PaperlessBilling'] = "Yes" if st.checkbox("Dijital Fatura (Kağıtsız)", value=True) else "No"
        
        payment_map = {
            "Elektronik Çek": "Electronic check",
            "Posta Çeki": "Mailed check",
            "Banka Transferi (Otomatik)": "Bank transfer (automatic)",
            "Kredi Kartı (Otomatik)": "Credit card (automatic)"
        }
        payment_ui = st.selectbox("Ödeme Yöntemi", list(payment_map.keys()))
        input_data['PaymentMethod'] = payment_map[payment_ui]
        
        monthly_charges = st.number_input("Aylık Fatura Tutarı ($)", 18.0, 150.0, 70.0)
        input_data['MonthlyCharges'] = monthly_charges
        input_data['TotalCharges'] = monthly_charges * tenure

    # Tahmin Butonu
    predict_btn = st.sidebar.button("🚀 Risk Analizini Başlat", type="primary", use_container_width=True)

    if predict_btn:
        # --- VERİ HAZIRLIĞI ---
        try:
            df = pd.DataFrame([input_data])
            
            # Label Encoding
            for col in categorical_cols:
                if col in df.columns:
                    le = label_encoders[col]
                    # Bilinmeyen label gelirse handle et
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
            
            # Sütun sırasını garantiye al
            df = df[feature_names]
            
            # --- TAHMİN ---
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
            
            # --- SONUÇ EKRANI ---
            
            # Sekmeler
            tab1, tab2, tab3 = st.tabs(["📊 Analiz Özeti", "🧠 Yapay Zeka Görüşü (SHAP)", "⚡ What-If Simülasyonu"])
            
            with tab1:
                col_res1, col_res2 = st.columns([1, 1.5])
                
                with col_res1:
                    # Gösterge
                    st.plotly_chart(create_gauge(probability), use_container_width=True)
                    
                    # Risk Kartı
                    risk_color = "#e74c3c" if probability > 0.5 else "#2ecc71"
                    risk_text = "YÜKSEK RİSK" if probability > 0.5 else "DÜŞÜK RİSK"
                    
                    st.markdown(f"""
                    <div style="background-color: {risk_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: white; margin:0;">{risk_text}</h3>
                        <p style="margin:0; font-size: 14px;">Churn Olasılığı: %{probability*100:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_res2:
                    st.subheader("💡 Aksiyon Planı")
                    recommendations = get_smart_recommendations(probability, input_data)
                    
                    for title, desc in recommendations:
                        st.markdown(f"""
                        <div class="metric-card" style="margin-bottom: 10px; padding: 10px; border-left: 4px solid #3498db;">
                            <strong>{title}</strong><br>
                            <span style="color: #7f8c8d; font-size: 0.9em;">{desc}</span>
                        </div>
                        """, unsafe_allow_html=True)

            with tab2:
                st.subheader("Model Kararını Etkileyen Faktörler")
                st.info("Bu grafik, modelin neden bu kararı verdiğini açıklar (SHAP Analizi).")
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df)
                
                # SHAP Görselleştirme (Custom Plotly)
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values[0]
                })
                shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values('Abs SHAP', ascending=True).tail(10)
                
                fig_shap = px.bar(
                    shap_df, x='SHAP Value', y='Feature', orientation='h',
                    color='SHAP Value',
                    color_continuous_scale=['#e74c3c', '#ecf0f1', '#2ecc71'],
                    title="En Etkili 10 Kriter"
                )
                fig_shap.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_shap, use_container_width=True)
                
            with tab3:
                st.subheader("🎲 Senaryo Analizi")
                st.markdown("Değişkenleri değiştirerek riski nasıl düşürebileceğinizi test edin.")
                
                col_sim1, col_sim2 = st.columns(2)
                with col_sim1:
                    new_monthly = st.slider("Yeni Aylık Ücret ($)", 18.0, 150.0, float(monthly_charges), key="sim_price")
                with col_sim2:
                    new_tenure = st.slider("Abonelik Süresini Uzat (+Ay)", 0, 24, 0, key="sim_tenure")
                
                if st.button("Senaryoyu Hesapla"):
                    # Basit simülasyon mantığı (Burada model tekrar çalıştırılabilir)
                    # Örnek olarak basit matematiksel bir yaklaşım gösteriyoruz, 
                    # gerçek uygulamada df kopyalanıp tekrar model.predict yapılmalı.
                    
                    sim_df = df.copy()
                    sim_df['MonthlyCharges'] = new_monthly
                    sim_df['tenure'] = df['tenure'] + new_tenure
                    
                    sim_prob = model.predict_proba(sim_df)[0][1]
                    diff = probability - sim_prob
                    
                    st.success(f"Bu değişiklikler ile risk **%{sim_prob*100:.1f}** seviyesine inebilir.")
                    st.metric("Risk Değişimi", f"%{sim_prob*100:.1f}", f"-{diff*100:.1f}%", delta_color="inverse")

        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
            st.warning("Lütfen model dosyalarının doğru yüklendiğinden emin olun.")
            
    else:
        # Başlangıç Ekranı
        st.info("👈 Analize başlamak için sol menüden müşteri bilgilerini girip butona tıklayın.")
        
        # Dashboard boşken şık görünsün diye dummy metrics
        col1, col2, col3 = st.columns(3)
        metrics = [
            ("Ortalama Müşteri Ömrü", "32 Ay", "+2.4%"),
            ("Aylık Churn Oranı", "%12.4", "-1.2%"),
            ("Aktif Müşteri", "4,245", "+84")
        ]
        
        for col, (label, val, delta) in zip([col1, col2, col3], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color: #7f8c8d;">{label}</h4>
                    <h2 style="margin:10px 0; color: #2c3e50;">{val}</h2>
                    <span style="color: #27ae60; font-weight:bold;">{delta}</span> <span style="font-size:0.8em">geçen aya göre</span>
                </div>
                """, unsafe_allow_html=True)
                
        # [Image of data visualization dashboard concept] 
        # Not: Yukarıdaki metrikler görsel amaçlıdır, gerçek veri tabanına bağlı değildir.

if __name__ == "__main__":
    main()