# Laporan Proyek Machine Learning - Sistem Rekomendasi Penjualan Mobil Bekas

## Project Overview

Industri penjualan mobil bekas merupakan sektor yang terus berkembang di seluruh dunia. Dengan banyaknya pilihan mobil bekas yang tersedia di pasaran, konsumen sering kali kesulitan untuk menemukan mobil yang sesuai dengan preferensi dan kebutuhan mereka. Di sisi lain, penjual mobil bekas juga menghadapi tantangan dalam memasarkan produk mereka kepada calon pembeli yang tepat.

Sistem rekomendasi dapat menjadi solusi untuk masalah ini dengan membantu konsumen menemukan mobil bekas yang sesuai dengan preferensi mereka dan membantu penjual untuk menargetkan calon pembeli potensial. Dengan memanfaatkan data historis penjualan mobil bekas, sistem rekomendasi dapat memberikan saran yang personal dan relevan kepada konsumen, meningkatkan pengalaman berbelanja, dan pada akhirnya meningkatkan konversi penjualan.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi penjualan mobil bekas yang dapat membantu konsumen menemukan mobil yang sesuai dengan preferensi mereka berdasarkan berbagai faktor seperti merek, model, tahun, harga, dan fitur lainnya.

## Business Understanding

Pada bagian ini, akan dijelaskan proses klarifikasi masalah dalam konteks bisnis penjualan mobil bekas.

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah rumusan masalah yang akan diselesaikan dalam proyek ini:

1. Bagaimana cara mengembangkan sistem rekomendasi yang dapat membantu konsumen menemukan mobil bekas yang sesuai dengan preferensi mereka?
2. Bagaimana cara memanfaatkan data historis penjualan mobil bekas untuk memberikan rekomendasi yang personal dan relevan?
3. Bagaimana cara mengevaluasi efektivitas sistem rekomendasi yang dikembangkan?

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan sistem rekomendasi yang dapat memberikan saran mobil bekas yang sesuai dengan preferensi konsumen.
2. Memanfaatkan data historis penjualan mobil bekas untuk menghasilkan rekomendasi yang personal dan relevan.
3. Mengevaluasi efektivitas sistem rekomendasi dengan menggunakan metrik evaluasi yang sesuai.

### Solution Statements

Untuk mencapai tujuan di atas, berikut adalah pendekatan yang akan digunakan:

1. **Content-Based Filtering**: Mengembangkan sistem rekomendasi berbasis konten yang merekomendasikan mobil bekas berdasarkan kesamaan fitur dengan mobil yang pernah dilihat atau diminati oleh konsumen. Pendekatan ini akan menggunakan teknik seperti cosine similarity untuk mengukur kesamaan antar mobil.

2. **Collaborative Filtering**: Mengembangkan sistem rekomendasi berbasis kolaboratif yang merekomendasikan mobil bekas berdasarkan preferensi konsumen lain yang memiliki selera serupa. Pendekatan ini akan menggunakan teknik seperti matrix factorization atau nearest neighbors untuk mengidentifikasi pola preferensi konsumen.

## Data Understanding

Pada proyek ini, kita menggunakan dua dataset penjualan mobil bekas yang tersedia di folder data yaitu `car__sales__data.csv` dan `Updated_Car_Sales_Data.csv`. Dataset ini berisi informasi tentang mobil bekas yang telah terjual, termasuk merek, model, tahun, jarak tempuh, harga, jenis bahan bakar, warna, transmisi, fitur, kondisi, dan riwayat kecelakaan. Dataset ini diambil dari Kaggle dan dapat diakses di [Link ini](https://www.kaggle.com/datasets/benjnb/cars-for-sale/data?select=Updated_Car_Sales_Data.csv).

Variabel-variabel pada dataset mobil bekas adalah sebagai berikut:
- **Car Make**: Merek mobil (contoh: Honda, Toyota, BMW)
- **Car Model**: Model mobil (contoh: Accord, Camry, X5)
- **Year**: Tahun pembuatan mobil
- **Mileage**: Jarak tempuh mobil dalam mil
- **Price**: Harga mobil dalam USD
- **Fuel Type**: Jenis bahan bakar (Gasoline, Diesel, Electric, Hybrid)
- **Color**: Warna mobil
- **Transmission**: Jenis transmisi (Automatic, Manual)
- **Options/Features**: Fitur-fitur yang tersedia pada mobil
- **Condition**: Kondisi mobil (New, Used, Like New)
- **Accident**: Riwayat kecelakaan (Yes, No)

Berikut adalah beberapa insight dari analisis eksplorasi data:

1. Dataset original memiliki 20.000 data, sementara dataset updated memiliki 23.000 data.
2. Tidak ada nilai yang hilang (missing values) pada kedua dataset.
3. Rentang tahun mobil adalah dari 2010 hingga 2022 pada dataset original, dan 2010 hingga 2023 pada dataset updated.
4. Harga mobil berkisar antara $4.001 hingga $49.998 pada dataset original, dan $4.001 hingga $299.922 pada dataset updated.
5. Jarak tempuh mobil berkisar antara 5.015 mil hingga 149.987 mil.
6. Terdapat 20 merek mobil pada dataset original dan 29 merek pada dataset updated.
7. Terdapat 4 jenis bahan bakar pada dataset original (Gasoline, Diesel, Electric, Hybrid) dan 5 jenis pada dataset updated.
8. Mayoritas mobil memiliki transmisi otomatis.
9. Mayoritas mobil memiliki riwayat kecelakaan (Yes).

## Data Preparation

Berikut adalah tahapan data preparation yang dilakukan dalam proyek ini:

1. **Penggabungan Dataset**: Menggabungkan dataset original dan updated untuk mendapatkan dataset yang lebih komprehensif.

2. **Penanganan Nilai Duplikat**: Memeriksa dan menghapus data duplikat jika ada untuk menghindari bias dalam rekomendasi.

3. **Feature Engineering**:
   - Membuat fitur baru 'Car_Age' berdasarkan tahun mobil untuk menangkap informasi tentang usia mobil.
   - Mengekstrak fitur dari kolom 'Options/Features' menjadi kolom-kolom terpisah untuk setiap fitur (seperti 'Bluetooth', 'GPS', 'Heated Seats', dll).

4. **Encoding Fitur Kategorikal**:
   - Menggunakan One-Hot Encoding untuk fitur kategorikal seperti 'Car Make', 'Car Model', 'Fuel Type', 'Color', 'Transmission', dan 'Condition'.
   - Mengubah fitur 'Accident' menjadi nilai numerik (Yes=1, No=0).

5. **Normalisasi Fitur Numerik**: Menggunakan StandardScaler untuk menormalisasi fitur numerik seperti 'Year', 'Price', 'Mileage', dan 'Car_Age' agar semua fitur memiliki skala yang sama dan tidak ada fitur yang mendominasi dalam perhitungan kesamaan.

Tahapan data preparation ini penting dilakukan untuk memastikan data siap digunakan dalam pembuatan model rekomendasi. Normalisasi fitur numerik diperlukan karena algoritma berbasis jarak seperti cosine similarity dan K-Nearest Neighbors sangat sensitif terhadap skala fitur. Encoding fitur kategorikal diperlukan untuk mengubah data kategorikal menjadi format numerik yang dapat diproses oleh algoritma machine learning.

## Modeling

Pada proyek ini, dua pendekatan utama digunakan untuk membangun sistem rekomendasi mobil bekas:

### 1. Content-Based Filtering

Pendekatan content-based filtering merekomendasikan mobil berdasarkan kesamaan fitur dengan mobil yang pernah dilihat atau diminati oleh pengguna. Implementasi pendekatan ini melibatkan langkah-langkah berikut:

1. **Persiapan Fitur**: Semua fitur yang telah diproses pada tahap data preparation digunakan untuk membentuk representasi vektor dari setiap mobil.

2. **Perhitungan Kesamaan**: Menggunakan cosine similarity untuk menghitung kesamaan antara setiap pasangan mobil dalam dataset. Cosine similarity mengukur sudut kosinus antara dua vektor, dengan nilai yang lebih tinggi menunjukkan kesamaan yang lebih besar.

3. **Fungsi Rekomendasi**: Membuat fungsi `get_content_based_recommendations` yang menerima indeks mobil dan mengembalikan daftar mobil yang paling mirip berdasarkan skor kesamaan.

Contoh output rekomendasi berbasis konten:
```
Mobil yang dipilih:
Car Make         Land Rover
Car Model       Range Rover
Year                   2016
Price              25414.06
Mileage              115056
Fuel Type            Diesel
Transmission         Manual

Rekomendasi Mobil Serupa (Berbasis Konten):
     Car Make    Car Model  Year      Price  Mileage Fuel Type Transmission
4  Land Rover  Range Rover  2019  40919.460   147954  Gasoline       Manual
1  Land Rover  Range Rover  2015  30266.950   116241    Diesel       Manual
2  Land Rover  Range Rover  2013  24334.216   127637    Diesel       Manual
0  Land Rover       Evoque  2017  10994.704   110364    Diesel       Manual
3  Land Rover  Range Rover  2015   8001.104    80677  Gasoline       Manual
```

### 2. K-Nearest Neighbors (KNN)

Pendekatan KNN digunakan sebagai alternatif untuk content-based filtering. KNN mencari mobil yang paling mirip berdasarkan jarak dalam ruang fitur. Implementasi pendekatan ini melibatkan langkah-langkah berikut:

1. **Inisialisasi Model**: Membuat model NearestNeighbors dengan parameter n_neighbors=6 (5 rekomendasi + 1 mobil yang dipilih) dan metric='euclidean'.

2. **Pelatihan Model**: Melatih model dengan data fitur yang telah dinormalisasi.

3. **Fungsi Rekomendasi**: Membuat fungsi `get_knn_recommendations` yang menerima indeks mobil dan mengembalikan daftar mobil yang paling dekat berdasarkan jarak Euclidean.

Kelebihan dan kekurangan dari pendekatan yang dipilih:

**Content-Based Filtering**:
- Kelebihan: Tidak memerlukan data pengguna lain, dapat memberikan rekomendasi untuk item baru, dan dapat memberikan penjelasan mengapa item tertentu direkomendasikan.
- Kekurangan: Cenderung merekomendasikan item yang sangat mirip (overspecialization), tidak dapat menemukan preferensi baru yang mungkin disukai pengguna.

**K-Nearest Neighbors**:
- Kelebihan: Sederhana dan intuitif, dapat menangkap hubungan kompleks antar fitur, dan tidak memerlukan asumsi tentang distribusi data.
- Kekurangan: Sensitif terhadap skala fitur, komputasi dapat menjadi mahal untuk dataset besar, dan pemilihan nilai k yang optimal dapat menjadi tantangan.

## Evaluation

Untuk mengevaluasi efektivitas sistem rekomendasi yang dikembangkan, beberapa metrik evaluasi digunakan:

### 1. Precision@k

Precision@k mengukur proporsi item yang relevan dari k item yang direkomendasikan. Dalam konteks sistem rekomendasi mobil bekas, precision@k dapat diinterpretasikan sebagai proporsi mobil yang benar-benar sesuai dengan preferensi pengguna dari k mobil yang direkomendasikan.

Formula:
```
Precision@k = (Jumlah item relevan dalam k rekomendasi) / k
```

### 2. Recall@k

Recall@k mengukur proporsi item relevan yang berhasil direkomendasikan dari total item relevan yang tersedia. Dalam konteks sistem rekomendasi mobil bekas, recall@k dapat diinterpretasikan sebagai proporsi mobil yang sesuai dengan preferensi pengguna yang berhasil direkomendasikan dari total mobil yang sesuai dengan preferensi pengguna.

Formula:
```
Recall@k = (Jumlah item relevan dalam k rekomendasi) / (Total jumlah item relevan)
```

### 3. Cosine Similarity Score

Untuk content-based filtering, cosine similarity score digunakan untuk mengukur kesamaan antara mobil yang dipilih dan mobil yang direkomendasikan. Skor yang lebih tinggi menunjukkan kesamaan yang lebih besar.

Formula:
```
Cosine Similarity(A, B) = (A · B) / (||A|| * ||B||)
```
dimana A dan B adalah vektor fitur dari dua mobil, A · B adalah dot product, dan ||A|| dan ||B|| adalah norma dari vektor A dan B.

### 4. Euclidean Distance

Untuk pendekatan KNN, Euclidean distance digunakan untuk mengukur jarak antara mobil yang dipilih dan mobil yang direkomendasikan. Jarak yang lebih kecil menunjukkan kesamaan yang lebih besar.

Formula:
```
Euclidean Distance(A, B) = sqrt(sum((A_i - B_i)^2))
```
dimana A dan B adalah vektor fitur dari dua mobil, dan A_i dan B_i adalah nilai fitur ke-i.

### Hasil Evaluasi

Berdasarkan evaluasi yang dilakukan, kedua pendekatan (content-based filtering dan KNN) menunjukkan kinerja yang baik dalam memberikan rekomendasi mobil bekas yang relevan. Content-based filtering cenderung memberikan rekomendasi mobil dengan merek dan model yang sama dengan mobil yang dipilih, sementara KNN dapat memberikan rekomendasi yang lebih beragam tetapi tetap relevan.

Pendekatan hybrid yang menggabungkan kedua metode dapat memberikan hasil yang lebih baik dengan memanfaatkan kelebihan dari masing-masing pendekatan. Misalnya, content-based filtering dapat digunakan untuk memberikan rekomendasi awal, kemudian KNN dapat digunakan untuk memperluas rekomendasi dengan mobil yang mungkin tidak terlalu mirip tetapi tetap relevan dengan preferensi pengguna.


