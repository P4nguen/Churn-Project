import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    """Telco Customer Churn verisini yükle ve hazırla"""
    # Kaggle'dan indirilen veriyi yükle
    # URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # CustomerID'yi çıkar
    df = df.drop('customerID', axis=1)
    
    # TotalCharges'ı numeric yap (boşlukları temizle)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Churn'u binary yap
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def encode_features(df):
    """Kategorik değişkenleri encode et"""
    # Kategorik kolonları belirle
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders, categorical_cols

def train_model(df):
    """CatBoost modeli eğit"""
    # X ve y'yi ayır
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # CatBoost modeli
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=False
    )
    
    # Modeli eğit
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    
    # Test performansı
    from sklearn.metrics import classification_report, roc_auc_score
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Model Performansı:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return model, X.columns.tolist()

def save_model(model, label_encoders, feature_names, categorical_cols):
    """Modeli ve yardımcı objeleri kaydet"""
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    with open('categorical_cols.pkl', 'wb') as f:
        pickle.dump(categorical_cols, f)
    
    print("\nModel ve yardımcı dosyalar kaydedildi!")
    print("- churn_model.pkl")
    print("- label_encoders.pkl")
    print("- feature_names.pkl")
    print("- categorical_cols.pkl")

if __name__ == "__main__":
    print("Veri yükleniyor...")
    df = prepare_data()
    
    print("Özellikler encode ediliyor...")
    df_encoded, label_encoders, categorical_cols = encode_features(df)
    
    print("Model eğitiliyor...")
    model, feature_names = train_model(df_encoded)
    
    print("Model kaydediliyor...")
    save_model(model, label_encoders, feature_names, categorical_cols)
    
    print("\n✅ Tüm işlemler tamamlandı! Artık 'streamlit run app.py' komutuyla uygulamayı başlatabilirsiniz.")