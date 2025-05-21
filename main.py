# Sistem Rekomendasi Penjualan Mobil Bekas
# Proyek Machine Learning

# Import library yang diperlukan
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
import re 
import joblib
import os
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')
sns.set_style('whitegrid')

os.makedirs('models', exist_ok=True)

def load_data():
    """
    Memuat dataset penjualan mobil bekas
    
    Returns:
        tuple: Dataset original dan dataset updated
    """
    df_original = pd.read_csv('./data/car__sales__data.csv')
    df_updated = pd.read_csv('./data/Updated_Car_Sales_Data.csv')
    return df_original, df_updated


def extract_features(features_str):
    """
    Mengekstrak fitur dari string fitur yang dipisahkan koma
    
    Args:
        features_str (str): String fitur yang dipisahkan koma
        
    Returns:
        list: Daftar fitur yang diekstrak
    """
    if pd.isna(features_str):
        return []
    
    features_str = str(features_str).lower()
    features = re.split(r'[,;/|]', features_str)
    features = [f.strip() for f in features]
    return features


def preprocess_data(df):
    """
    Melakukan pra-pemrosesan data untuk sistem rekomendasi
    
    Args:
        df (DataFrame): DataFrame yang akan diproses
        
    Returns:
        tuple: DataFrame fitur, DataFrame informasi mobil, dan matriks kesamaan
    """
    # Ekstraksi fitur dari kolom Options/Features
    df['Extracted_Features'] = df['Options/Features'].apply(extract_features)
    
    # Mendapatkan fitur yang paling umum
    all_features = []
    for features in df['Extracted_Features']:
        all_features.extend(features)
    
    feature_counts = pd.Series(all_features).value_counts()
    top_features = feature_counts.head(20).index.tolist()
    
    # Membuat kolom one-hot untuk fitur teratas
    for feature in top_features:
        df[f'has_{feature.replace(" ", "_")}'] = df['Extracted_Features'].apply(lambda x: 1 if feature in x else 0)
    
    # Konversi kolom kategorikal menjadi numerik 
    categorical_cols = ['Car Make', 'Car Model', 'Fuel Type', 'Color', 'Transmission', 'Condition']
    numerical_cols = ['Year', 'Mileage', 'Price']
    
    # Membuat fitur baru: Usia mobil
    current_year = 2025
    df['Car_Age'] = current_year - df['Year']
    
    # One-hot encoding untuk kolom kategorikal
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    
    # Menyimpan informasi mobil untuk rekomendasi
    car_info = df[['Car Make', 'Car Model', 'Year', 'Price', 'Mileage', 'Fuel Type', 'Transmission']]
    
    # Menghapus kolom yang tidak diperlukan lagi
    columns_to_drop = categorical_cols + ['Options/Features', 'Extracted_Features']
    features_df = df.drop(columns=columns_to_drop)
    
    # Konversi kolom Accident menjadi numerik
    features_df['Accident'] = features_df['Accident'].map({'Yes': 1, 'No': 0})
    
    # Normalisasi fitur numerik
    scaler = StandardScaler()
    numerical_features = ['Year', 'Price', 'Mileage', 'Car_Age']
    features_df[numerical_features] = scaler.fit_transform(features_df[numerical_features])
    
    # Menghitung matriks kesamaan
    similarity_matrix = cosine_similarity(features_df)
    
    return features_df, car_info, similarity_matrix


def get_content_based_recommendations(car_index, similarity_matrix, car_info, n=5, asc=True):
    """
    Mendapatkan rekomendasi mobil berdasarkan kesamaan konten
    
    Args:
        car_index (int): Indeks mobil yang dipilih
        similarity_matrix (ndarray): Matriks kesamaan
        car_info (DataFrame): DataFrame informasi mobil
        n (int): Jumlah rekomendasi yang diinginkan
        asc (bool): Mengurutkan berdasarkan harga (True: ascending, False: descending)
        
    Returns:
        DataFrame: Rekomendasi mobil
    """
    # Mendapatkan skor kesamaan untuk mobil yang dipilih
    similarity_scores = list(enumerate(similarity_matrix[car_index]))
    
    # Mengurutkan mobil berdasarkan skor kesamaan 
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Mendapatkan skor n mobil teratas (kecuali mobil itu sendiri)
    similarity_scores = similarity_scores[1:n+1]
    
    # Mendapatkan indeks mobil
    car_indices = [i[0] for i in similarity_scores]
    
    return car_info.iloc[car_indices].reset_index(drop=True).sort_values('Price', ascending=asc)


def build_knn_model(features_df, n_neighbors=6):
    """
    Membangun model KNN untuk rekomendasi
    
    Args:
        features_df (DataFrame): DataFrame fitur
        n_neighbors (int): Jumlah tetangga terdekat
        
    Returns:
        NearestNeighbors: Model KNN
    """
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
    knn_model.fit(features_df)
    return knn_model


def get_knn_recommendations(car_index, model, car_info, features_df, n=5, asc=True):
    """
    Mendapatkan rekomendasi mobil menggunakan model KNN
    
    Args:
        car_index (int): Indeks mobil yang dipilih
        model (NearestNeighbors): Model KNN
        car_info (DataFrame): DataFrame informasi mobil
        features_df (DataFrame): DataFrame fitur
        n (int): Jumlah rekomendasi yang diinginkan
        asc (bool): Mengurutkan berdasarkan harga (True: ascending, False: descending)
        
    Returns:
        DataFrame: Rekomendasi mobil
    """
    # Mendapatkan fitur mobil yang dipilih
    car_features = features_df.iloc[car_index].values.reshape(1, -1)
    
    # Menemukan tetangga terdekat
    distances, indices = model.kneighbors(car_features)
    
    # Mendapatkan indeks mobil (kecuali mobil itu sendiri)
    car_indices = indices.flatten()[1:n+1]
    
    # Mengembalikan informasi mobil yang direkomendasikan
    return car_info.iloc[car_indices].reset_index(drop=True).sort_values('Price', ascending=asc)


def get_knowledge_based_recommendations(car_info, make, model=None, max_price=None, min_year=None, fuel_type=None, n=5, asc=True):
    """
    Mendapatkan rekomendasi mobil berdasarkan kriteria spesifik
    
    Args:
        car_info (DataFrame): DataFrame informasi mobil
        make (str): Merek mobil
        model (str, optional): Model mobil
        max_price (float, optional): Harga maksimum
        min_year (int, optional): Tahun minimum
        fuel_type (str, optional): Jenis bahan bakar
        n (int): Jumlah rekomendasi yang diinginkan
        asc (bool): Mengurutkan berdasarkan harga (True: ascending, False: descending)
        
    Returns:
        DataFrame: Rekomendasi mobil
    """
    # Filter berbasis merek
    recommendations = car_info[car_info['Car Make'] == make].copy()
    
    # Filter berdasrkan model jika ditentukan
    if model:
        recommendations = recommendations[recommendations['Car Model'] == model]
    
    # Fitur berdasarkan harga maksimum jika ditentukan
    if max_price:
        recommendations = recommendations[recommendations['Price'] <= max_price]
    
    # Fitur berdasarkan minimum jika ditentukan
    if min_year:
        recommendations = recommendations[recommendations['Year'] >= min_year]
    
    # Filter berdasarkan jenis bahan bakar jika ditentukan
    if fuel_type:
        recommendations = recommendations[recommendations['Fuel Type'] == fuel_type]
        
    recommendations = recommendations.sort_values('Price', ascending=asc).head(n)
    
    return recommendations.reset_index(drop=True)


def calculate_recommendation_similarity(selected_car, recommended_cars, numerical_features=['Year', 'Price', 'Mileage']):
    """
    Menghitung kesamaan antara mobil yang dipilih dan mobil yang direkomendasikan
    
    Args:
        selected_car (Series): Mobil yang dipilih
        recommended_cars (DataFrame): Mobil yang direkomendasikan
        numerical_features (list): Fitur numerik untuk perbandingan
        
    Returns:
        float: Skor kesamaan
    """
    # Menghitung rata-rata perbedaan relatif untuk fitur numerik
    similarity_scores = []
    
    for feature in numerical_features:
        selected_value = selected_car[feature]
        recommended_values = recommended_cars[feature].values
        
        # Menghindari pembagian dengan nol
        if selected_value == 0:
            continue
            
        # Menghitung perbedaan relatif
        relative_diff = np.abs(recommended_values - selected_value) / selected_value
        similarity_scores.append(1 - np.mean(relative_diff))
    
    # Mengembalikan rata-rata skor kesamaan
    return np.mean(similarity_scores) if similarity_scores else 0


def save_model(model, filepath='./models/knn_model.joblib'):
    """
    Menyimpan model ke file
    
    Args:
        model: Model yang akan disimpan
        filepath (str): Jalur file untuk menyimpan model
    """
    joblib.dump(model, filepath)


def load_model(filepath='./models/knn_model.joblib'):
    """
    Memuat model dari file
    
    Args:
        filepath (str): Jalur file model
        
    Returns:
        Model yang dimuat
    """
    return joblib.load(filepath)


def main():
    """
    Fungsi utama untuk menjalankan sistem rekomendasi
    """
    print("Memuat data...")
    df_original, df_updated = load_data()
    
    # Menggunakan dataset yang diperbarui untuk analisis
    df = df_updated.copy()
    
    print("Melakukan pra-pemrosesan data...")
    features_df, car_info, similarity_matrix = preprocess_data(df)
    
    print("Membangun model KNN...")
    knn_model = build_knn_model(features_df)
    
    # Contoh penggunaan sistem rekomendasi
    sample_car_index = 1  # Indeks mobil contoh
    
    print("\nMobil yang dipilih:")
    print(car_info.iloc[sample_car_index])
    
    print("\nRekomendasi Mobil Serupa (Berbasis Konten):")
    content_recommendations = get_content_based_recommendations(sample_car_index, similarity_matrix, car_info, asc=False)
    print(content_recommendations)
    
    print("\nRekomendasi Mobil Serupa (KNN):")
    knn_recommendations = get_knn_recommendations(sample_car_index, knn_model, car_info, features_df, asc=False)
    print(knn_recommendations)
    
    print("\nRekomendasi Mobil Berbasis Kriteria (Berbasis Pengetahuan):")
    knowledge_recommendations = get_knowledge_based_recommendations(car_info, make='Honda', max_price=25000, min_year=2015, asc=True, n=10)
    print(knowledge_recommendations)
    
    # Menyimpan model untuk penggunaan di masa mendatang
    save_model(knn_model)
    
    print("\nModel KNN telah disimpan di ./models/knn_model.joblib")


if __name__ == "__main__":
    main()