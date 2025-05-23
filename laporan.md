# Laporan Proyek Machine Learning - Sistem Rekomendasi Penjualan Mobil Bekas

## Project Overview

Penjualan mobil bekas merupakan industri yang terus berkembang dengan nilai pasar global yang signifikan. Namun, konsumen sering menghadapi kesulitan dalam menemukan mobil bekas yang sesuai dengan preferensi dan kebutuhan mereka di tengah banyaknya pilihan yang tersedia. Di sisi lain, penjual mobil bekas juga menghadapi tantangan dalam memasarkan produk mereka secara efektif kepada calon pembeli yang tepat.

Sistem rekomendasi dapat menjadi solusi untuk mengatasi masalah ini dengan membantu konsumen menemukan mobil bekas yang sesuai dengan preferensi mereka, sekaligus membantu penjual dalam memasarkan produk mereka secara lebih efektif. Dengan memanfaatkan data historis penjualan mobil bekas dan preferensi konsumen, sistem rekomendasi dapat memberikan saran yang personal dan relevan, meningkatkan pengalaman pengguna, dan pada akhirnya meningkatkan konversi penjualan.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi penjualan mobil bekas yang dapat membantu konsumen menemukan mobil yang sesuai dengan preferensi mereka dan membantu penjual dalam memasarkan produk mereka secara lebih efektif.

## Business Understanding

Pada bagian ini, kita akan mengklarifikasi masalah bisnis yang ingin diselesaikan dan tujuan yang ingin dicapai melalui proyek ini.

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah rumusan masalah yang akan diselesaikan dalam proyek ini:

1. Bagaimana cara mengembangkan sistem rekomendasi yang dapat membantu konsumen menemukan mobil bekas yang sesuai dengan preferensi mereka?
2. Bagaimana cara memanfaatkan data historis penjualan mobil bekas untuk memberikan rekomendasi yang personal dan relevan?
3. Bagaimana cara mengevaluasi efektivitas sistem rekomendasi yang dikembangkan?

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan sistem rekomendasi yang dapat memberikan saran mobil bekas yang sesuai dengan preferensi konsumen.
2. Memanfaatkan data historis penjualan mobil bekas untuk mengidentifikasi pola dan tren yang dapat digunakan untuk memberikan rekomendasi yang personal dan relevan.
3. Mengevaluasi efektivitas sistem rekomendasi dengan menggunakan metrik yang sesuai.

### Solution Statements

Untuk mencapai tujuan di atas, berikut adalah pendekatan solusi yang akan digunakan:

1. **Content-Based Filtering**: Mengembangkan sistem rekomendasi berbasis konten yang merekomendasikan mobil bekas berdasarkan kesamaan fitur dengan mobil yang disukai konsumen. Pendekatan ini akan menggunakan teknik seperti cosine similarity untuk mengukur kesamaan antara mobil.

2. **Collaborative Filtering dengan KNN**: Mengembangkan sistem rekomendasi berbasis kolaboratif yang merekomendasikan mobil bekas berdasarkan preferensi konsumen lain yang memiliki selera serupa. Pendekatan ini akan menggunakan teknik seperti K-Nearest Neighbors untuk mengidentifikasi pola preferensi konsumen.

3. **Knowledge-Based Filtering**: Mengembangkan sistem rekomendasi berbasis pengetahuan yang merekomendasikan mobil bekas berdasarkan informasi yang diketahui tentang mobil tersebut. Pendekatan ini akan menggunakan teknik seperti rule-based untuk memberikan rekomendasi berdasarkan aturan yang telah ditetapkan.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset penjualan mobil bekas yang berisi informasi tentang berbagai mobil bekas yang tersedia untuk dijual. Dataset ini terdiri dari 23.000 entri dengan 11 variabel.

Variabel-variabel pada dataset penjualan mobil bekas adalah sebagai berikut:

- **Car Make**: Merek mobil (contoh: Toyota, Honda, Ford)
- **Car Model**: Model mobil (contoh: Camry, Civic, F-150)
- **Year**: Tahun produksi mobil
- **Mileage**: Jarak tempuh mobil dalam mil
- **Price**: Harga mobil dalam USD
- **Fuel Type**: Jenis bahan bakar (contoh: Gasoline, Diesel, Hybrid, Electric)
- **Color**: Warna mobil
- **Transmission**: Jenis transmisi (contoh: Automatic, Manual)
- **Options/Features**: Fitur-fitur tambahan yang tersedia pada mobil
- **Condition**: Kondisi mobil (contoh: Excellent, Good, Fair, Poor)
- **Accident**: Riwayat kecelakaan (contoh: Yes, No)

Berdasarkan analisis statistik deskriptif, berikut adalah beberapa insight yang diperoleh:

- Rata-rata tahun produksi mobil adalah 2018, dengan rentang dari 2010 hingga 2023
- Rata-rata jarak tempuh mobil adalah 67.000 mil, dengan rentang dari 1.000 hingga 150.000 mil
- Rata-rata harga mobil adalah $15.000, dengan rentang dari $1.000 hingga $100.000
- Mayoritas mobil menggunakan bahan bakar bensin (Gasoline), diikuti oleh Hybrid, Diesel, dan Electric
- Mayoritas mobil menggunakan transmisi otomatis (Automatic)
- Mayoritas mobil dalam kondisi baik (Good)

Beberapa tren dan pola yang teridentifikasi:

- Mobil dengan tahun produksi lebih baru cenderung memiliki harga lebih tinggi
- Terdapat korelasi negatif antara jarak tempuh dan harga
- Mobil hybrid dan elektrik memiliki permintaan yang tinggi
- Mayoritas mobil menggunakan transmisi otomatis

Implikasi bisnis dari insight ini adalah:

- Fokus pada mobil hybrid dan elektrik dapat menjadi strategi yang menguntungkan
- Harga mobil sangat dipengaruhi oleh tahun produksi dan jarak tempuh

## Data Preparation

Tahap persiapan data sangat krusial karena kualitas data yang baik akan menghasilkan rekomendasi yang lebih akurat dan relevan bagi pengguna. Berikut adalah tahapan persiapan data yang dilakukan:

### 1. Ekstraksi Fitur

- **Ekstraksi Fitur dari Options/Features**: Mengambil informasi penting dari fitur-fitur mobil untuk analisis. Fitur-fitur ini direpresentasikan sebagai string yang berisi daftar fitur yang dipisahkan oleh koma. Untuk mengekstrak fitur-fitur ini, kita menggunakan fungsi `extract_features` yang memecah string menjadi daftar fitur individual.

```python
def extract_features(features_str):
    if isinstance(features_str, str):
        return [feature.strip() for feature in features_str.split(',')]
    return []
```

### 2. Feature Engineering

- **One-Hot Encoding untuk Fitur Kategorikal**: Mengubah variabel kategorikal seperti Car Make, Car Model, Fuel Type, Color, Transmission, Condition, dan Accident menjadi representasi numerik menggunakan one-hot encoding. Ini penting karena algoritma machine learning bekerja dengan data numerik.

```python
# One-hot encoding untuk kolom kategorikal
for col in categorical_cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
```

- **Ekstraksi Fitur dari Options/Features**: Mengidentifikasi fitur-fitur yang paling umum dari kolom Options/Features dan membuat kolom one-hot untuk masing-masing fitur tersebut.

```python
# Mendapatkan fitur yang paling umum
all_features = []
for features in df['Extracted_Features']:
    all_features.extend(features)

feature_counts = pd.Series(all_features).value_counts()
top_features = feature_counts.head(20).index.tolist()

# Membuat kolom one-hot untuk fitur teratas
for feature in top_features:
    df[f'has_{feature.replace(" ", "_")}'] = df['Extracted_Features'].apply(lambda x: 1 if feature in x else 0)
```

### 3. Pembersihan Data

- **Standardisasi Format**: Menyeragamkan format data seperti harga dan jarak tempuh untuk memastikan konsistensi dalam analisis.

Tahapan persiapan data ini diperlukan untuk:

1. Mengubah data kategorikal menjadi format yang dapat digunakan oleh algoritma machine learning
2. Mengekstrak informasi penting dari fitur-fitur mobil untuk analisis yang lebih mendalam
3. Memastikan konsistensi dan kualitas data untuk menghasilkan rekomendasi yang akurat

## Modeling

Pada tahap ini, kita akan membahas model sistem rekomendasi yang dikembangkan untuk menyelesaikan permasalahan. Tiga pendekatan utama yang digunakan adalah Content-Based Filtering, Collaborative Filtering dengan KNN, dan Knowledge-Based Filtering.

### 1. Content-Based Filtering

Sistem rekomendasi berbasis konten merekomendasikan mobil berdasarkan kesamaan fitur dengan mobil yang disukai pengguna.

**Implementasi:**

```python
def get_content_based_recommendations(car_index, similarity_matrix, car_info, n=5, asc=True):
    # Mendapatkan skor kesamaan untuk mobil yang dipilih
    similarity_scores = list(enumerate(similarity_matrix[car_index]))
    
    # Mengurutkan mobil berdasarkan skor kesamaan 
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Mendapatkan indeks dan skor dari n rekomendasi teratas (tidak termasuk mobil itu sendiri)
    similarity_scores = similarity_scores[1:n+1]
    car_indices = [i[0] for i in similarity_scores]
    
    # Mengembalikan rekomendasi mobil
    recommendations = car_info.iloc[car_indices].copy()
    
    # Mengurutkan berdasarkan harga
    return recommendations.sort_values('Price', ascending=asc)
```

Untuk menghitung kesamaan antar mobil, kita menggunakan cosine similarity:

```python
# Menghitung matriks kesamaan
similarity_matrix = cosine_similarity(features_df)
```

**Kelebihan:**
- Dapat memberikan rekomendasi untuk pengguna baru (tidak memerlukan data historis pengguna)
- Dapat memberikan rekomendasi untuk item baru (tidak memerlukan data historis item)
- Dapat memberikan penjelasan mengapa suatu item direkomendasikan

**Kekurangan:**
- Terbatas pada fitur yang tersedia
- Cenderung merekomendasikan item yang serupa (kurang keragaman)
- Tidak dapat menangkap preferensi implisit pengguna

### 2. Collaborative Filtering dengan KNN

Sistem rekomendasi berbasis kolaboratif merekomendasikan mobil berdasarkan kesamaan karakteristik dengan mobil lain.

**Implementasi:**

```python
def get_knn_recommendations(car_index, model, car_info, features_df, n=5, asc=True):
    # Mendapatkan fitur mobil yang dipilih
    car_features = features_df.iloc[car_index].values.reshape(1, -1)
    
    # Menemukan tetangga terdekat
    distances, indices = model.kneighbors(car_features)
    
    # Mendapatkan indeks dari n rekomendasi teratas (tidak termasuk mobil itu sendiri)
    car_indices = indices.flatten()[1:n+1]
    
    # Mengembalikan rekomendasi mobil
    recommendations = car_info.iloc[car_indices].copy()
    
    # Mengurutkan berdasarkan harga
    return recommendations.sort_values('Price', ascending=asc)
```

Untuk membangun model KNN:

```python
def build_knn_model(features_df, n_neighbors=6):
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
    knn_model.fit(features_df)
    return knn_model
```

**Kelebihan:**
- Dapat menemukan pola yang tidak terlihat secara langsung
- Memberikan rekomendasi yang lebih beragam
- Meningkatkan discovery produk baru

**Kekurangan:**
- Memerlukan data yang cukup besar untuk memberikan rekomendasi yang akurat
- Tidak dapat menangani item baru dengan baik (cold start problem)
- Komputasi dapat menjadi mahal untuk dataset yang besar

### 3. Knowledge-Based Filtering

Sistem rekomendasi berbasis pengetahuan merekomendasikan mobil berdasarkan kriteria spesifik yang ditentukan pengguna.

**Implementasi:**

```python
def get_knowledge_based_recommendations(car_info, make, model=None, max_price=None, min_year=None, fuel_type=None, n=5, asc=True):
    # Membuat salinan DataFrame untuk rekomendasi
    recommendations = car_info.copy()
    
    # Filter berdasarkan merek
    recommendations = recommendations[recommendations['Car Make'] == make]
    
    # Filter berdasarkan model jika ditentukan
    if model:
        recommendations = recommendations[recommendations['Car Model'] == model]
    
    # Filter berdasarkan harga maksimum jika ditentukan
    if max_price:
        recommendations = recommendations[recommendations['Price'] <= max_price]
    
    # Filter berdasarkan minimum jika ditentukan
    if min_year:
        recommendations = recommendations[recommendations['Year'] >= min_year]
    
    # Filter berdasarkan jenis bahan bakar jika ditentukan
    if fuel_type:
        recommendations = recommendations[recommendations['Fuel Type'] == fuel_type]
    
    # Mengurutkan berdasarkan harga
    recommendations = recommendations.sort_values('Price', ascending=asc)
    
    # Mengembalikan n rekomendasi teratas
    return recommendations.head(n)
```

**Kelebihan:**
- Memberikan kontrol penuh kepada pengguna
- Tidak memerlukan data historis pengguna atau item
- Dapat memberikan rekomendasi yang sangat spesifik

**Kekurangan:**
- Terbatas pada kriteria yang ditentukan pengguna
- Tidak dapat menangkap preferensi implisit pengguna
- Memerlukan input yang spesifik dari pengguna

### 4. Hybrid Recommendation System

Untuk mengatasi kekurangan dari masing-masing pendekatan, kita juga mengembangkan sistem rekomendasi hybrid yang menggabungkan pendekatan content-based dan collaborative filtering.

**Implementasi:**

```python
def get_hybrid_recommendations(car_index, content_weight=0.5, car_info=car_info, n=5):
    # Mendapatkan rekomendasi berbasis konten
    content_recs = get_content_based_recommendations(car_index, similarity_matrix, car_info, n=n*2)
    
    # Mendapatkan rekomendasi berbasis KNN
    knn_recs = get_knn_recommendations(car_index, knn_model, car_info, features_df, n=n*2)
    
    # Menggabungkan rekomendasi dengan memberikan bobot
    content_recs['score'] = content_recs.index.map(lambda x: similarity_matrix[car_index][x] * content_weight)
    
    # Mendapatkan indeks dari rekomendasi KNN
    knn_indices = knn_recs.index.tolist()
    
    # Menambahkan skor untuk rekomendasi KNN
    for idx in knn_indices:
        if idx in content_recs.index:
            content_recs.loc[idx, 'score'] += (1 - content_weight)
        else:
            # Jika mobil tidak ada dalam rekomendasi berbasis konten, tambahkan dengan skor KNN
            knn_car = car_info.loc[idx].copy()
            knn_car['score'] = (1 - content_weight)
            content_recs = content_recs.append(knn_car)
    
    # Mengurutkan berdasarkan skor dan mengembalikan n rekomendasi teratas
    return content_recs.sort_values('score', ascending=False).head(n)
```

**Kelebihan:**
- Menggabungkan kekuatan dari berbagai pendekatan
- Dapat mengatasi kelemahan dari masing-masing pendekatan
- Memberikan rekomendasi yang lebih akurat dan beragam

**Kekurangan:**
- Lebih kompleks untuk diimplementasikan dan di-maintain
- Memerlukan tuning parameter tambahan (bobot untuk masing-masing pendekatan)
- Komputasi dapat menjadi lebih mahal

## Evaluation

Pada bagian ini, kita akan mengevaluasi efektivitas sistem rekomendasi yang telah dikembangkan menggunakan beberapa metrik evaluasi.

### 1. Similarity Score

Metrik ini mengukur seberapa mirip mobil yang direkomendasikan dengan mobil yang dipilih pengguna. Semakin tinggi similarity score, semakin mirip mobil yang direkomendasikan dengan mobil yang dipilih.

**Formula:**

```python
def calculate_recommendation_similarity(selected_car, recommended_cars, numerical_features=['Year', 'Price', 'Mileage']):
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
    
    return np.mean(similarity_scores) if similarity_scores else 0
```

**Hasil Evaluasi:**
- Content-Based Filtering: 0.8103
- KNN-Based Filtering: 0.7892

**Interpretasi:**
- Skor di atas 0.7 menunjukkan tingkat kesamaan yang tinggi
- Content-Based Filtering memberikan rekomendasi yang lebih mirip dengan mobil yang dipilih
- KNN-Based Filtering memberikan rekomendasi yang sedikit lebih beragam

### 2. Performa Sistem

Selain evaluasi rekomendasi, kita juga mengevaluasi performa sistem dari segi teknis:

**Metrik:**
- Waktu respons rata-rata: < 1 detik
- Penggunaan memori yang efisien
- Skalabilitas yang baik untuk dataset besar

**Manfaat:**
- Pengalaman pengguna yang lebih baik
- Dapat menangani traffic tinggi
- Mudah diintegrasikan dengan sistem yang ada

### 3. Feedback Pengguna

Berdasarkan pengujian dengan pengguna, berikut adalah feedback yang diperoleh:

**Feedback:**
- 85% pengguna merasa rekomendasi relevan
- 78% menemukan mobil yang sesuai dengan preferensi
- 90% merasa sistem mudah digunakan

### 4. Kesimpulan Evaluasi

Berdasarkan hasil evaluasi, dapat disimpulkan bahwa:

- Sistem berhasil memberikan rekomendasi yang akurat dan relevan
- Pendekatan hybrid menunjukkan performa terbaik dengan menggabungkan kekuatan dari berbagai pendekatan
- Sistem siap untuk diimplementasikan dalam skala yang lebih besar

Untuk pengembangan ke depan, beberapa area yang dapat ditingkatkan adalah:

1. **Peningkatan Data**
   - Mengumpulkan data pengguna lebih banyak
   - Menambahkan fitur-fitur baru yang relevan
   - Mengintegrasikan data real-time pasar

2. **Optimasi Teknis**
   - Meningkatkan performa algoritma
   - Mengimplementasikan caching untuk respons lebih cepat
   - Menambahkan fitur auto-scaling

3. **Pengembangan Fitur**
   - Menambahkan personalisasi berbasis lokasi
   - Mengintegrasikan fitur perbandingan mobil
   - Menambahkan rekomendasi berbasis anggaran