#!/usr/bin/env python3
"""
Sistem Rekomendasi Penjualan Mobil Bekas
========================================

Script ini mengimplementasikan tiga jenis sistem rekomendasi:
1. Content-Based Filtering
2. Collaborative Filtering (KNN)
3. Hybrid Recommendation System

Author: Data Science Team
Date: 2024
"""

# Import library yang diperlukan
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

class CarRecommendationSystem:
    """
    Sistem Rekomendasi Mobil Bekas
    """
    
    def __init__(self, data_path='./data/car_listings_cleaned.csv'):
        """
        Inisialisasi sistem rekomendasi
        
        Parameters:
        -----------
        data_path : str
            Path ke file dataset
        """
        self.data_path = data_path
        self.df_original = None
        self.df_model = None
        self.features_scaled = None
        self.cosine_sim = None
        self.knn_model = None
        self.scaler = None
        self.le_brand = None
        self.le_transmission = None
        self.le_fuel = None
        self.features_for_similarity = ['price', 'mileage', 'extracted_year', 'engine_size', 
                                      'brand_encoded', 'transmission_encoded', 'fuel_type_encoded']
        
    def load_data(self):
        """
        Memuat dataset
        """
        print("Loading dataset...")
        self.df_original = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df_original.shape}")
        return self.df_original
    
    def extract_car_info(self, name):
        """
        Ekstraksi informasi mobil dari kolom name
        
        Parameters:
        -----------
        name : str
            Nama lengkap mobil
            
        Returns:
        --------
        tuple
            (year, engine_size, transmission, fuel_type)
        """
        # Extract year (4 digits)
        year_match = re.search(r'\b(19|20)\d{2}\b', str(name))
        year = int(year_match.group()) if year_match else None
        
        # Extract engine size (decimal number followed by A or L)
        engine_match = re.search(r'(\d+\.?\d*)[AL]', str(name))
        engine_size = float(engine_match.group(1)) if engine_match else None
        
        # Extract transmission type
        transmission = 'Automatic' if 'A' in str(name) else 'Manual'
        
        # Extract fuel type
        fuel_type = 'Hybrid' if 'Hybrid' in str(name) else 'Petrol'
        if 'Diesel' in str(name):
            fuel_type = 'Diesel'
        elif 'Electric' in str(name):
            fuel_type = 'Electric'
        
        return year, engine_size, transmission, fuel_type
    
    def preprocess_data(self):
        """
        Preprocessing data
        """
        print("Preprocessing data...")
        
        # Membuat salinan dataframe
        self.df_model = self.df_original.copy()
        
        # Ekstraksi informasi dari kolom 'name'
        extracted_info = self.df_model['name'].apply(
            lambda x: pd.Series(self.extract_car_info(x))
        )
        self.df_model[['extracted_year', 'engine_size', 'transmission', 'fuel_type']] = extracted_info
        
        # Clean dan prepare data
        self.df_model['price'] = pd.to_numeric(self.df_model['price'], errors='coerce')
        self.df_model['depreciation'] = pd.to_numeric(self.df_model['depreciation'], errors='coerce')
        self.df_model['mileage'] = pd.to_numeric(self.df_model['mileage'], errors='coerce')
        
        # Handle missing values
        self.df_model['owners'].fillna(self.df_model['owners'].median(), inplace=True)
        self.df_model['depreciation'].fillna(self.df_model['depreciation'].median(), inplace=True)
        self.df_model['extracted_year'].fillna(self.df_model['extracted_year'].median(), inplace=True)
        self.df_model['engine_size'].fillna(self.df_model['engine_size'].median(), inplace=True)
        
        # Encoding categorical variables
        self.le_brand = LabelEncoder()
        self.le_transmission = LabelEncoder()
        self.le_fuel = LabelEncoder()
        
        self.df_model['brand_encoded'] = self.le_brand.fit_transform(self.df_model['brand'])
        self.df_model['transmission_encoded'] = self.le_transmission.fit_transform(self.df_model['transmission'])
        self.df_model['fuel_type_encoded'] = self.le_fuel.fit_transform(self.df_model['fuel_type'])
        
        # Membuat fitur numerik untuk similarity calculation
        df_features = self.df_model[self.features_for_similarity].fillna(0)
        
        # Standardize features
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(df_features)
        
        print(f"Preprocessing completed. Features shape: {self.features_scaled.shape}")
        
    def build_content_based_model(self):
        """
        Membangun model Content-Based Filtering
        """
        print("Building Content-Based model...")
        self.cosine_sim = cosine_similarity(self.features_scaled)
        print(f"Cosine similarity matrix shape: {self.cosine_sim.shape}")
        
    def build_collaborative_model(self):
        """
        Membangun model Collaborative Filtering (KNN)
        """
        print("Building Collaborative Filtering model...")
        self.knn_model = NearestNeighbors(n_neighbors=6, algorithm='auto', metric='euclidean')
        self.knn_model.fit(self.features_scaled)
        print("KNN model trained successfully")
        
    def get_content_based_recommendations(self, car_index, n=5, asc=True):
        """
        Mendapatkan rekomendasi berdasarkan Content-Based Filtering
        
        Parameters:
        -----------
        car_index : int
            Index mobil referensi
        n : int
            Jumlah rekomendasi
        asc : bool
            Urutan berdasarkan harga
            
        Returns:
        --------
        DataFrame
            Rekomendasi mobil
        """
        # Mendapatkan skor kesamaan untuk mobil yang dipilih
        similarity_scores = list(enumerate(self.cosine_sim[car_index]))
        
        # Mengurutkan mobil berdasarkan skor kesamaan 
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Mendapatkan skor n mobil teratas (kecuali mobil itu sendiri)
        similarity_scores = similarity_scores[1:n+1]
        
        # Mendapatkan indeks mobil
        car_indices = [i[0] for i in similarity_scores]
        
        # Mengembalikan informasi mobil yang direkomendasikan
        result = self.df_model.iloc[car_indices][['name', 'brand', 'price', 'mileage', 'extracted_year', 
                                                'engine_size', 'transmission', 'fuel_type']].copy()
        result['similarity_score'] = [i[1] for i in similarity_scores]
        
        return result.sort_values('price', ascending=asc)
    
    def get_knn_recommendations(self, car_index, n=5, asc=True):
        """
        Mendapatkan rekomendasi berdasarkan KNN
        
        Parameters:
        -----------
        car_index : int
            Index mobil referensi
        n : int
            Jumlah rekomendasi
        asc : bool
            Urutan berdasarkan harga
            
        Returns:
        --------
        DataFrame
            Rekomendasi mobil
        """
        # Mendapatkan fitur mobil yang dipilih
        car_features = self.features_scaled[car_index].reshape(1, -1)
        
        # Menemukan tetangga terdekat
        distances, indices = self.knn_model.kneighbors(car_features)
        
        # Mendapatkan indeks mobil (kecuali mobil itu sendiri)
        car_indices = indices.flatten()[1:n+1]
        
        # Mengembalikan informasi mobil yang direkomendasikan
        result = self.df_model.iloc[car_indices][['name', 'brand', 'price', 'mileage', 'extracted_year', 
                                                'engine_size', 'transmission', 'fuel_type']].copy()
        result['distance'] = distances.flatten()[1:n+1]
        
        return result.sort_values('price', ascending=asc)
    
    def get_knowledge_based_recommendations(self, brand=None, max_price=None, min_year=None, 
                                          max_mileage=None, fuel_type=None, 
                                          transmission=None, n=5, asc=True):
        """
        Sistem rekomendasi berbasis pengetahuan
        
        Parameters:
        -----------
        brand : str, optional
            Merek mobil yang diinginkan
        max_price : float, optional
            Harga maksimum yang diinginkan
        min_year : int, optional
            Tahun minimum produksi
        max_mileage : int, optional
            Jarak tempuh maksimum
        fuel_type : str, optional
            Jenis bahan bakar
        transmission : str, optional
            Jenis transmisi
        n : int
            Jumlah rekomendasi yang diinginkan
        asc : bool
            Urutan berdasarkan harga
        
        Returns:
        --------
        DataFrame
            Mobil yang direkomendasikan berdasarkan kriteria
        """
        # Mulai dengan semua data
        recommendations = self.df_model.copy()
        
        # Filter berdasarkan kriteria
        if brand:
            recommendations = recommendations[recommendations['brand'].str.lower() == brand.lower()]
        
        if max_price:
            recommendations = recommendations[recommendations['price'] <= max_price]
        
        if min_year:
            recommendations = recommendations[recommendations['extracted_year'] >= min_year]
        
        if max_mileage:
            recommendations = recommendations[recommendations['mileage'] <= max_mileage]
        
        if fuel_type:
            recommendations = recommendations[recommendations['fuel_type'].str.lower() == fuel_type.lower()]
        
        if transmission:
            recommendations = recommendations[recommendations['transmission'].str.lower() == transmission.lower()]
        
        # Jika tidak ada hasil, kembalikan DataFrame kosong
        if recommendations.empty:
            print("Tidak ada mobil yang sesuai dengan kriteria yang diberikan.")
            return pd.DataFrame()
        
        # Urutkan berdasarkan harga dan ambil n teratas
        recommendations = recommendations.sort_values('price', ascending=asc).head(n)
        
        # Pilih kolom yang relevan untuk ditampilkan
        result_columns = ['name', 'brand', 'price', 'mileage', 'extracted_year', 
                         'engine_size', 'transmission', 'fuel_type']
        
        return recommendations[result_columns].reset_index(drop=True)
    
    def get_hybrid_recommendations(self, car_index, content_weight=0.6, knn_weight=0.4, n=5, asc=True):
        """
        Sistem rekomendasi hybrid
        
        Parameters:
        -----------
        car_index : int
            Index mobil yang dijadikan referensi
        content_weight : float
            Bobot untuk Content-Based Filtering
        knn_weight : float
            Bobot untuk Collaborative Filtering
        n : int
            Jumlah rekomendasi yang diinginkan
        asc : bool
            Urutan berdasarkan harga
        
        Returns:
        --------
        DataFrame
            Mobil yang direkomendasikan dengan hybrid score
        """
        # Mendapatkan rekomendasi dari Content-Based Filtering
        content_recs = self.get_content_based_recommendations(car_index, n=n*2)
        
        # Mendapatkan rekomendasi dari Collaborative Filtering (KNN)
        knn_recs = self.get_knn_recommendations(car_index, n=n*2)
        
        # Membuat dictionary untuk menyimpan scores
        hybrid_scores = {}
        
        # Proses Content-Based recommendations
        for idx, row in content_recs.iterrows():
            car_name = row['name']
            similarity_score = row.get('similarity_score', 0)
            hybrid_scores[car_name] = {
                'content_score': similarity_score,
                'knn_score': 0,
                'data': row
            }
        
        # Proses KNN recommendations
        for idx, row in knn_recs.iterrows():
            car_name = row['name']
            # Convert distance to similarity (lower distance = higher similarity)
            distance = row.get('distance', 1)
            knn_similarity = 1 / (1 + distance)
            
            if car_name in hybrid_scores:
                hybrid_scores[car_name]['knn_score'] = knn_similarity
            else:
                hybrid_scores[car_name] = {
                    'content_score': 0,
                    'knn_score': knn_similarity,
                    'data': row
                }
        
        # Hitung hybrid score
        for car_name in hybrid_scores:
            content_score = hybrid_scores[car_name]['content_score']
            knn_score = hybrid_scores[car_name]['knn_score']
            
            # Weighted combination
            hybrid_score = (content_weight * content_score) + (knn_weight * knn_score)
            hybrid_scores[car_name]['hybrid_score'] = hybrid_score
        
        # Convert ke DataFrame
        hybrid_results = []
        for car_name, scores in hybrid_scores.items():
            row_data = scores['data'].copy()
            row_data['content_score'] = scores['content_score']
            row_data['knn_score'] = scores['knn_score']
            row_data['hybrid_score'] = scores['hybrid_score']
            hybrid_results.append(row_data)
        
        hybrid_df = pd.DataFrame(hybrid_results)
        
        # Sort berdasarkan hybrid score dan ambil top n
        hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False).head(n)
        
        # Sort berdasarkan harga jika diminta
        if asc is not None:
            hybrid_df = hybrid_df.sort_values('price', ascending=asc)
        
        return hybrid_df[['name', 'brand', 'price', 'mileage', 'extracted_year', 
                         'engine_size', 'transmission', 'fuel_type', 
                         'content_score', 'knn_score', 'hybrid_score']]
    
    def evaluate_content_based_similarity(self, sample_index, n_recommendations=5):
        """
        Evaluasi similarity untuk Content-Based Filtering
        """
        print("=== EVALUASI CONTENT-BASED FILTERING ===")
        
        # Dapatkan rekomendasi
        content_recs = self.get_content_based_recommendations(sample_index, n=n_recommendations)
        
        # Informasi mobil yang dipilih
        selected_car = self.df_model.iloc[sample_index]
        print(f"Mobil yang dipilih: {selected_car['name']}")
        print(f"Brand: {selected_car['brand']}, Harga: ${selected_car['price']:,.2f}")
        
        # Analisis similarity scores
        if 'similarity_score' in content_recs.columns:
            avg_similarity = content_recs['similarity_score'].mean()
            min_similarity = content_recs['similarity_score'].min()
            max_similarity = content_recs['similarity_score'].max()
            
            print("\nSimilarity Scores:")
            print(f"- Average: {avg_similarity:.4f}")
            print(f"- Minimum: {min_similarity:.4f}")
            print(f"- Maximum: {max_similarity:.4f}")
            print(f"- Standard Deviation: {content_recs['similarity_score'].std():.4f}")
        
        # Analisis karakteristik rekomendasi
        same_brand = (content_recs['brand'] == selected_car['brand']).sum()
        same_transmission = (content_recs['transmission'] == selected_car['transmission']).sum()
        same_fuel = (content_recs['fuel_type'] == selected_car['fuel_type']).sum()
        
        print("\nAnalisis Karakteristik:")
        print(f"- Mobil dengan brand sama: {same_brand}/{len(content_recs)} ({same_brand/len(content_recs)*100:.1f}%)")
        print(f"- Mobil dengan transmisi sama: {same_transmission}/{len(content_recs)} ({same_transmission/len(content_recs)*100:.1f}%)")
        print(f"- Mobil dengan bahan bakar sama: {same_fuel}/{len(content_recs)} ({same_fuel/len(content_recs)*100:.1f}%)")
        
        # Analisis harga
        price_diff = abs(content_recs['price'] - selected_car['price'])
        avg_price_diff = price_diff.mean()
        price_similarity = 1 - (price_diff / selected_car['price']).mean()
        
        print("\nAnalisis Harga:")
        print(f"- Rata-rata selisih harga: ${avg_price_diff:,.2f}")
        print(f"- Price similarity score: {price_similarity:.4f}")
        
        return content_recs, avg_similarity if 'similarity_score' in content_recs.columns else price_similarity
    
    def evaluate_collaborative_filtering(self, sample_index, n_recommendations=5):
        """
        Evaluasi similarity untuk Collaborative Filtering (KNN)
        """
        print("=== EVALUASI COLLABORATIVE FILTERING (KNN) ===")
        
        # Dapatkan rekomendasi
        knn_recs = self.get_knn_recommendations(sample_index, n=n_recommendations)
        
        # Informasi mobil yang dipilih
        selected_car = self.df_model.iloc[sample_index]
        print(f"Mobil yang dipilih: {selected_car['name']}")
        print(f"Brand: {selected_car['brand']}, Harga: ${selected_car['price']:,.2f}")
        
        # Analisis distance scores
        if 'distance' in knn_recs.columns:
            avg_distance = knn_recs['distance'].mean()
            min_distance = knn_recs['distance'].min()
            max_distance = knn_recs['distance'].max()
            
            # Convert distance to similarity (lower distance = higher similarity)
            similarity_scores = 1 / (1 + knn_recs['distance'])
            avg_similarity = similarity_scores.mean()
            
            print("\nDistance Scores:")
            print(f"- Average distance: {avg_distance:.4f}")
            print(f"- Minimum distance: {min_distance:.4f}")
            print(f"- Maximum distance: {max_distance:.4f}")
            print(f"- Average similarity: {avg_similarity:.4f}")
        
        # Analisis karakteristik rekomendasi
        same_brand = (knn_recs['brand'] == selected_car['brand']).sum()
        same_transmission = (knn_recs['transmission'] == selected_car['transmission']).sum()
        same_fuel = (knn_recs['fuel_type'] == selected_car['fuel_type']).sum()
        
        print("\nAnalisis Karakteristik:")
        print(f"- Mobil dengan brand sama: {same_brand}/{len(knn_recs)} ({same_brand/len(knn_recs)*100:.1f}%)")
        print(f"- Mobil dengan transmisi sama: {same_transmission}/{len(knn_recs)} ({same_transmission/len(knn_recs)*100:.1f}%)")
        print(f"- Mobil dengan bahan bakar sama: {same_fuel}/{len(knn_recs)} ({same_fuel/len(knn_recs)*100:.1f}%)")
        
        # Analisis harga
        price_diff = abs(knn_recs['price'] - selected_car['price'])
        avg_price_diff = price_diff.mean()
        price_similarity = 1 - (price_diff / selected_car['price']).mean()
        
        print("\nAnalisis Harga:")
        print(f"- Rata-rata selisih harga: ${avg_price_diff:,.2f}")
        print(f"- Price similarity score: {price_similarity:.4f}")
        
        return knn_recs, avg_similarity if 'distance' in knn_recs.columns else price_similarity
    
    def evaluate_hybrid_recommendations(self, car_index, content_weight=0.6, knn_weight=0.4, n=5):
        """
        Evaluasi sistem rekomendasi hybrid
        """
        print("=== EVALUASI SISTEM REKOMENDASI HYBRID ===")
        
        # Dapatkan rekomendasi hybrid
        hybrid_recs = self.get_hybrid_recommendations(car_index, content_weight, knn_weight, n=n)
        
        # Informasi mobil yang dipilih
        selected_car = self.df_model.iloc[car_index]
        print(f"Mobil yang dipilih: {selected_car['name']}")
        print(f"Brand: {selected_car['brand']}, Harga: ${selected_car['price']:,.2f}")
        
        # Analisis hybrid scores
        avg_hybrid_score = hybrid_recs['hybrid_score'].mean()
        avg_content_score = hybrid_recs['content_score'].mean()
        avg_knn_score = hybrid_recs['knn_score'].mean()
        
        print("\nHybrid Scores:")
        print(f"- Average Hybrid Score: {avg_hybrid_score:.4f}")
        print(f"- Average Content Score: {avg_content_score:.4f}")
        print(f"- Average KNN Score: {avg_knn_score:.4f}")
        print(f"- Content Weight: {content_weight}")
        print(f"- KNN Weight: {knn_weight}")
        
        # Analisis karakteristik
        same_brand = (hybrid_recs['brand'] == selected_car['brand']).sum()
        same_transmission = (hybrid_recs['transmission'] == selected_car['transmission']).sum()
        same_fuel = (hybrid_recs['fuel_type'] == selected_car['fuel_type']).sum()
        
        print("\nAnalisis Karakteristik:")
        print(f"- Mobil dengan brand sama: {same_brand}/{len(hybrid_recs)} ({same_brand/len(hybrid_recs)*100:.1f}%)")
        print(f"- Mobil dengan transmisi sama: {same_transmission}/{len(hybrid_recs)} ({same_transmission/len(hybrid_recs)*100:.1f}%)")
        print(f"- Mobil dengan bahan bakar sama: {same_fuel}/{len(hybrid_recs)} ({same_fuel/len(hybrid_recs)*100:.1f}%)")
        
        # Analisis keragaman
        brand_diversity = hybrid_recs['brand'].nunique()
        price_range = hybrid_recs['price'].max() - hybrid_recs['price'].min()
        
        print("\nAnalisis Keragaman:")
        print(f"- Jumlah brand berbeda: {brand_diversity}")
        print(f"- Range harga: ${price_range:,.2f}")
        
        return hybrid_recs, avg_hybrid_score
    
    def compare_all_systems(self, car_index, n=5):
        """
        Membandingkan semua sistem rekomendasi
        """
        print("=" * 80)
        print("PERBANDINGAN SEMUA SISTEM REKOMENDASI")
        print("=" * 80)
        
        # Content-Based
        content_recs, content_sim = self.evaluate_content_based_similarity(car_index, n)
        
        print("\n" + "=" * 80)
        
        # Collaborative
        knn_recs, knn_sim = self.evaluate_collaborative_filtering(car_index, n)
        
        print("\n" + "=" * 80)
        
        # Hybrid
        hybrid_recs, hybrid_sim = self.evaluate_hybrid_recommendations(car_index, n=n)
        
        print("\n" + "=" * 80)
        print("RINGKASAN PERBANDINGAN SEMUA SISTEM")
        print("=" * 80)
        
        print(f"Content-Based Similarity: {content_sim:.4f}")
        print(f"Collaborative Similarity: {knn_sim:.4f}")
        print(f"Hybrid Similarity: {hybrid_sim:.4f}")
        
        # Analisis keragaman
        content_diversity = content_recs['brand'].nunique()
        knn_diversity = knn_recs['brand'].nunique()
        hybrid_diversity = hybrid_recs['brand'].nunique()
        
        print("\nKeragaman Brand:")
        print(f"- Content-Based: {content_diversity} brand")
        print(f"- Collaborative: {knn_diversity} brand")
        print(f"- Hybrid: {hybrid_diversity} brand")
        
        # Visualisasi perbandingan
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        systems = ['Content-Based', 'Collaborative', 'Hybrid']
        similarities = [content_sim, knn_sim, hybrid_sim]
        plt.bar(systems, similarities, color=['skyblue', 'lightgreen', 'orange'])
        plt.title('Similarity Score Comparison')
        plt.ylabel('Similarity Score')
        plt.ylim(0, 1)
        
        plt.subplot(1, 3, 2)
        diversities = [content_diversity, knn_diversity, hybrid_diversity]
        plt.bar(systems, diversities, color=['skyblue', 'lightgreen', 'orange'])
        plt.title('Brand Diversity Comparison')
        plt.ylabel('Number of Different Brands')
        
        plt.subplot(1, 3, 3)
        # Price range comparison
        content_price_range = content_recs['price'].max() - content_recs['price'].min()
        knn_price_range = knn_recs['price'].max() - knn_recs['price'].min()
        hybrid_price_range = hybrid_recs['price'].max() - hybrid_recs['price'].min()
        
        price_ranges = [content_price_range, knn_price_range, hybrid_price_range]
        plt.bar(systems, price_ranges, color=['skyblue', 'lightgreen', 'orange'])
        plt.title('Price Range Comparison')
        plt.ylabel('Price Range ($)')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'content_recs': content_recs,
            'knn_recs': knn_recs,
            'hybrid_recs': hybrid_recs,
            'similarities': similarities,
            'diversities': diversities
        }
    
    def visualize_data_analysis(self):
        """
        Visualisasi analisis data
        """
        print("Creating data visualizations...")
        
        # Visualisasi distribusi harga
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.df_model['price'].dropna(), bins=50, alpha=0.7, color='skyblue')
        plt.title('Distribusi Harga Mobil')
        plt.xlabel('Harga')
        plt.ylabel('Frekuensi')

        # Visualisasi distribusi jarak tempuh
        plt.subplot(1, 2, 2)
        plt.hist(self.df_model['mileage'], bins=50, alpha=0.7, color='lightgreen')
        plt.title('Distribusi Jarak Tempuh')
        plt.xlabel('Jarak Tempuh (km)')
        plt.ylabel('Frekuensi')

        plt.tight_layout()
        plt.show()
        
        # Visualisasi distribusi merek mobil
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        top_brands = self.df_model['brand'].value_counts().head(10)
        plt.bar(range(len(top_brands)), top_brands.values, color='skyblue')
        plt.title('Top 10 Merek Mobil')
        plt.xlabel('Merek')
        plt.ylabel('Jumlah')
        plt.xticks(range(len(top_brands)), top_brands.index, rotation=45)

        # Visualisasi distribusi jenis transmisi
        plt.subplot(2, 2, 2)
        transmission_counts = self.df_model['transmission'].value_counts()
        plt.pie(transmission_counts.values, labels=transmission_counts.index, autopct='%1.1f%%')
        plt.title('Distribusi Jenis Transmisi')

        # Visualisasi distribusi jenis bahan bakar
        plt.subplot(2, 2, 3)
        fuel_counts = self.df_model['fuel_type'].value_counts()
        plt.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%')
        plt.title('Distribusi Jenis Bahan Bakar')

        # Visualisasi hubungan harga vs jarak tempuh
        plt.subplot(2, 2, 4)
        plt.scatter(self.df_model['mileage'], self.df_model['price'], alpha=0.5, color='green')
        plt.title('Hubungan Harga vs Jarak Tempuh')
        plt.xlabel('Jarak Tempuh (km)')
        plt.ylabel('Harga')

        plt.tight_layout()
        plt.show()
    
    def save_models(self, model_dir='./models'):
        """
        Menyimpan model yang telah dilatih
        """
        # Buat direktori jika belum ada
        os.makedirs(model_dir, exist_ok=True)
        
        # Simpan model KNN
        joblib.dump(self.knn_model, os.path.join(model_dir, 'knn_model.joblib'))
        
        # Simpan scaler
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
        
        # Simpan label encoders
        joblib.dump(self.le_brand, os.path.join(model_dir, 'le_brand.joblib'))
        joblib.dump(self.le_transmission, os.path.join(model_dir, 'le_transmission.joblib'))
        joblib.dump(self.le_fuel, os.path.join(model_dir, 'le_fuel.joblib'))
        
        # Simpan cosine similarity matrix
        np.save(os.path.join(model_dir, 'cosine_similarity.npy'), self.cosine_sim)
        
        # Simpan features scaled
        np.save(os.path.join(model_dir, 'features_scaled.npy'), self.features_scaled)
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir='./models'):
        """
        Memuat model yang telah disimpan
        """
        # Load model KNN
        self.knn_model = joblib.load(os.path.join(model_dir, 'knn_model.joblib'))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        
        # Load label encoders
        self.le_brand = joblib.load(os.path.join(model_dir, 'le_brand.joblib'))
        self.le_transmission = joblib.load(os.path.join(model_dir, 'le_transmission.joblib'))
        self.le_fuel = joblib.load(os.path.join(model_dir, 'le_fuel.joblib'))
        
        # Load cosine similarity matrix
        self.cosine_sim = np.load(os.path.join(model_dir, 'cosine_similarity.npy'))
        
        # Load features scaled
        self.features_scaled = np.load(os.path.join(model_dir, 'features_scaled.npy'))
        
        print(f"Models loaded from {model_dir}")
    
    def train_all_models(self):
        """
        Melatih semua model
        """
        print("Training all recommendation models...")
        
        # Load dan preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Build models
        self.build_content_based_model()
        self.build_collaborative_model()
        
        print("All models trained successfully!")
    
    def get_car_info(self, car_index):
        """
        Mendapatkan informasi mobil berdasarkan index
        """
        if car_index >= len(self.df_model):
            print(f"Index {car_index} tidak valid. Dataset memiliki {len(self.df_model)} mobil.")
            return None
            
        car = self.df_model.iloc[car_index]
        return {
            'name': car['name'],
            'brand': car['brand'],
            'price': car['price'],
            'mileage': car['mileage'],
            'year': car['extracted_year'],
            'engine_size': car['engine_size'],
            'transmission': car['transmission'],
            'fuel_type': car['fuel_type']
        }
    
    def search_car_by_name(self, car_name):
        """
        Mencari mobil berdasarkan nama
        """
        matches = self.df_model[self.df_model['name'].str.contains(car_name, case=False, na=False)]
        if matches.empty:
            print(f"Tidak ditemukan mobil dengan nama '{car_name}'")
            return None
        
        print(f"Ditemukan {len(matches)} mobil:")
        for idx, row in matches.iterrows():
            print(f"Index {idx}: {row['name']} - ${row['price']:,.2f}")
        
        return matches.index.tolist()


def main():
    """
    Fungsi utama untuk menjalankan sistem rekomendasi
    """
    print("=" * 80)
    print("SISTEM REKOMENDASI PENJUALAN MOBIL BEKAS")
    print("=" * 80)
    
    # Inisialisasi sistem
    recommender = CarRecommendationSystem()
    
    # Train semua model
    recommender.train_all_models()
    
    # Visualisasi data
    recommender.visualize_data_analysis()
    
    # Test sistem dengan sample mobil
    sample_car_index = 0
    
    print(f"\n{'='*80}")
    print("TESTING SISTEM REKOMENDASI")
    print(f"{'='*80}")
    
    # Informasi mobil yang dipilih
    car_info = recommender.get_car_info(sample_car_index)
    print("Mobil yang dipilih:")
    print(f"- Nama: {car_info['name']}")
    print(f"- Brand: {car_info['brand']}")
    print(f"- Harga: ${car_info['price']:,.2f}")
    print(f"- Jarak Tempuh: {car_info['mileage']:,} km")
    print(f"- Tahun: {car_info['year']}")
    print(f"- Transmisi: {car_info['transmission']}")
    print(f"- Bahan Bakar: {car_info['fuel_type']}")
    
    # Test Content-Based Filtering
    print(f"\n{'='*80}")
    print("REKOMENDASI CONTENT-BASED FILTERING:")
    print(f"{'='*80}")
    content_recs = recommender.get_content_based_recommendations(sample_car_index, n=5)
    print(content_recs[['name', 'brand', 'price', 'similarity_score']].to_string(index=False))
    
    # Test Collaborative Filtering
    print(f"\n{'='*80}")
    print("REKOMENDASI COLLABORATIVE FILTERING (KNN):")
    print(f"{'='*80}")
    knn_recs = recommender.get_knn_recommendations(sample_car_index, n=5)
    print(knn_recs[['name', 'brand', 'price', 'distance']].to_string(index=False))
    
    # Test Knowledge-Based Filtering
    print(f"\n{'='*80}")
    print("REKOMENDASI KNOWLEDGE-BASED FILTERING:")
    print(f"{'='*80}")
    knowledge_recs = recommender.get_knowledge_based_recommendations(
        brand='Toyota', max_price=50000, min_year=2015, n=5
    )
    print(knowledge_recs.to_string(index=False))
    
    # Test Hybrid System
    print(f"\n{'='*80}")
    print("REKOMENDASI HYBRID SYSTEM:")
    print(f"{'='*80}")
    hybrid_recs = recommender.get_hybrid_recommendations(sample_car_index, n=5)
    print(hybrid_recs[['name', 'brand', 'price', 'hybrid_score']].to_string(index=False))
    
    # Evaluasi dan perbandingan semua sistem
    print(f"\n{'='*80}")
    print("EVALUASI DAN PERBANDINGAN SISTEM:")
    print(f"{'='*80}")
    recommender.compare_all_systems(sample_car_index)
    
    # Simpan model
    recommender.save_models()
    
    print(f"\n{'='*80}")
    print("SISTEM REKOMENDASI SIAP DIGUNAKAN!")
    print(f"{'='*80}")
    print("Fitur yang tersedia:")
    print("1. Content-Based Filtering")
    print("2. Collaborative Filtering (KNN)")
    print("3. Knowledge-Based Filtering")
    print("4. Hybrid Recommendation System")
    print("5. Model Evaluation & Comparison")
    print("6. Data Visualization")
    print("7. Model Persistence (Save/Load)")


def interactive_recommendation():
    """
    Fungsi untuk rekomendasi interaktif
    """
    print("=" * 80)
    print("SISTEM REKOMENDASI INTERAKTIF")
    print("=" * 80)
    
    # Load sistem yang sudah dilatih
    recommender = CarRecommendationSystem()
    
    try:
        # Coba load model yang sudah ada
        recommender.load_data()
        recommender.preprocess_data()
        recommender.load_models()
        print("Model berhasil dimuat!")
    except Exception as e:
        # Jika tidak ada model, latih dari awal
        print("Model tidak ditemukan. Melatih model baru...")
        recommender.train_all_models()
        recommender.save_models()
    
    while True:
        print("\n" + "=" * 50)
        print("PILIH JENIS REKOMENDASI:")
        print("=" * 50)
        print("1. Rekomendasi berdasarkan mobil tertentu (Content-Based)")
        print("2. Rekomendasi berdasarkan similarity (Collaborative)")
        print("3. Rekomendasi berdasarkan kriteria (Knowledge-Based)")
        print("4. Rekomendasi hybrid")
        print("5. Cari mobil berdasarkan nama")
        print("6. Lihat informasi mobil")
        print("7. Keluar")
        
        choice = input("\nPilih opsi (1-7): ").strip()
        
        if choice == '1':
            # Content-Based Recommendation
            try:
                car_index = int(input("Masukkan index mobil (0-{}): ".format(len(recommender.df_model)-1)))
                n_recs = int(input("Jumlah rekomendasi (default 5): ") or "5")
                
                car_info = recommender.get_car_info(car_index)
                if car_info:
                    print(f"\nMobil referensi: {car_info['name']}")
                    recs = recommender.get_content_based_recommendations(car_index, n=n_recs)
                    print("\nRekomendasi Content-Based:")
                    print(recs[['name', 'brand', 'price', 'similarity_score']].to_string(index=False))
            except ValueError:
                print("Input tidak valid!")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            # Collaborative Filtering
            try:
                car_index = int(input("Masukkan index mobil (0-{}): ".format(len(recommender.df_model)-1)))
                n_recs = int(input("Jumlah rekomendasi (default 5): ") or "5")
                
                car_info = recommender.get_car_info(car_index)
                if car_info:
                    print(f"\nMobil referensi: {car_info['name']}")
                    recs = recommender.get_knn_recommendations(car_index, n=n_recs)
                    print("\nRekomendasi Collaborative:")
                    print(recs[['name', 'brand', 'price', 'distance']].to_string(index=False))
            except ValueError:
                print("Input tidak valid!")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            # Knowledge-Based Recommendation
            print("\nMasukkan kriteria pencarian (kosongkan jika tidak ingin filter):")
            brand = input("Brand: ").strip() or None
            max_price = input("Harga maksimal: ").strip()
            max_price = float(max_price) if max_price else None
            min_year = input("Tahun minimal: ").strip()
            min_year = int(min_year) if min_year else None
            max_mileage = input("Jarak tempuh maksimal: ").strip()
            max_mileage = int(max_mileage) if max_mileage else None
            fuel_type = input("Jenis bahan bakar: ").strip() or None
            transmission = input("Jenis transmisi: ").strip() or None
            n_recs = int(input("Jumlah rekomendasi (default 5): ") or "5")
            
            recs = recommender.get_knowledge_based_recommendations(
                brand=brand, max_price=max_price, min_year=min_year,
                max_mileage=max_mileage, fuel_type=fuel_type,
                transmission=transmission, n=n_recs
            )
            
            if not recs.empty:
                print("\nRekomendasi Knowledge-Based:")
                print(recs.to_string(index=False))
            else:
                print("Tidak ada mobil yang sesuai kriteria!")
        
        elif choice == '4':
            # Hybrid Recommendation
            try:
                car_index = int(input("Masukkan index mobil (0-{}): ".format(len(recommender.df_model)-1)))
                content_weight = float(input("Bobot Content-Based (0-1, default 0.6): ") or "0.6")
                knn_weight = 1 - content_weight
                n_recs = int(input("Jumlah rekomendasi (default 5): ") or "5")
                
                car_info = recommender.get_car_info(car_index)
                if car_info:
                    print(f"\nMobil referensi: {car_info['name']}")
                    print(f"Bobot: Content-Based={content_weight}, Collaborative={knn_weight}")
                    recs = recommender.get_hybrid_recommendations(
                        car_index, content_weight=content_weight, 
                        knn_weight=knn_weight, n=n_recs
                    )
                    print("\nRekomendasi Hybrid:")
                    print(recs[['name', 'brand', 'price', 'hybrid_score']].to_string(index=False))
            except ValueError:
                print("Input tidak valid!")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            # Search by name
            car_name = input("Masukkan nama mobil yang dicari: ").strip()
            if car_name:
                recommender.search_car_by_name(car_name)
        
        elif choice == '6':
            # Show car info
            try:
                car_index = int(input("Masukkan index mobil (0-{}): ".format(len(recommender.df_model)-1)))
                car_info = recommender.get_car_info(car_index)
                if car_info:
                    print("\nInformasi Mobil:")
                    for key, value in car_info.items():
                        print(f"- {key.title()}: {value}")
            except ValueError:
                print("Input tidak valid!")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '7':
            print("Terima kasih telah menggunakan sistem rekomendasi!")
            break
        
        else:
            print("Pilihan tidak valid!")


class RecommendationAPI:
    """
    API wrapper untuk sistem rekomendasi
    """
    
    def __init__(self, model_dir='./models'):
        self.recommender = CarRecommendationSystem()
        self.model_dir = model_dir
        self._load_system()
    
    def _load_system(self):
        """Load sistem rekomendasi"""
        try:
            self.recommender.load_data()
            self.recommender.preprocess_data()
            self.recommender.load_models()
            print("Sistem rekomendasi berhasil dimuat!")
        except Exception as e:
            print(f"Error loading system: {e}")
            print("Melatih model baru...")
            self.recommender.train_all_models()
            self.recommender.save_models()
    
    def get_recommendations(self, method='hybrid', **kwargs):
        """
        Mendapatkan rekomendasi
        
        Parameters:
        -----------
        method : str
            Jenis rekomendasi ('content', 'collaborative', 'knowledge', 'hybrid')
        **kwargs : dict
            Parameter untuk masing-masing metode
        
        Returns:
        --------
        dict
            Hasil rekomendasi
        """
        try:
            if method == 'content':
                car_index = kwargs.get('car_index', 0)
                n = kwargs.get('n', 5)
                recs = self.recommender.get_content_based_recommendations(car_index, n=n)
                
            elif method == 'collaborative':
                car_index = kwargs.get('car_index', 0)
                n = kwargs.get('n', 5)
                recs = self.recommender.get_knn_recommendations(car_index, n=n)
                
            elif method == 'knowledge':
                recs = self.recommender.get_knowledge_based_recommendations(**kwargs)
                
            elif method == 'hybrid':
                car_index = kwargs.get('car_index', 0)
                content_weight = kwargs.get('content_weight', 0.6)
                knn_weight = kwargs.get('knn_weight', 0.4)
                n = kwargs.get('n', 5)
                recs = self.recommender.get_hybrid_recommendations(
                    car_index, content_weight=content_weight, 
                    knn_weight=knn_weight, n=n
                )
            
            else:
                return {'error': 'Invalid method'}
            
            return {
                'success': True,
                'method': method,
                'recommendations': recs.to_dict('records') if not recs.empty else [],
                'count': len(recs)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_car_info(self, car_index):
        """Mendapatkan informasi mobil"""
        try:
            car_info = self.recommender.get_car_info(car_index)
            return {
                'success': True,
                'car_info': car_info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_cars(self, query):
        """Mencari mobil berdasarkan nama"""
        try:
            matches = self.recommender.search_car_by_name(query)
            return {
                'success': True,
                'matches': matches,
                'count': len(matches) if matches else 0
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Mode interaktif
        interactive_recommendation()
    else:
        # Mode default - training dan testing
        main()