# IF3170 Artificial Intelligence Tugas Besar 2
## Scholar Classifier - Group 34 (EgiluyIzinnn🙏)

### Deskripsi Singkat
Repository ini berisi implementasi Machine Learning untuk membangun model klasifikasi dan memprediksi status akademik mahasiswa (Dropout, Enrolled, Graduate) berdasarkan dataset yang diberikan. Tahapan ini meliputi:
- **Data Cleaning**: Penanganan missing values, outlier handling (Log transformation, Clipping).
- **Feature Engineering**: Pembuatan fitur interaksi, binning (Age, Occupation, Qualification).
- **Preprocessing**: Feature Scaling (StandardScaler), Encoding (One-Hot & Ordinal), Normalization (Yeo-Johnson), dan Dimensionality Reduction (PCA/Feature Selection).
- **Imbalance Handling**: Menggunakan SMOTETomek untuk menangani ketidakseimbangan kelas pada data training.
- **Modeling**: Implementasi algoritma pembelajaran mesin (seperti DTL, Logistic Regression, SVM, dll).

### Cara Setup dan Run Program

#### 1. Prerequisites
Pastikan Anda telah menginstal Python (versi 3.8 atau lebih baru) dan package manager `pip`.

#### 2. Instalasi Dependencies
Jalankan perintah berikut untuk menginstal library yang dibutuhkan:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn
```

#### 3. Menjalankan Program
1. Clone repository ini ke local machine Anda.
2. Buka file notebook utama: `IF3170_Artificial_Intelligence___Tugas_Besar_2_Notebook.ipynb` menggunakan VS Code atau Jupyter Notebook.
3. Pastikan dataset (`train.csv` dan `test.csv`) berada di folder `data/`.
4. Jalankan sel (cells) secara berurutan dari atas ke bawah (Run All).

###  Pembagian Tugas Anggota Kelompok

Berikut adalah pembagian tugas untuk setiap anggota kelompok 34:

| Nama | NIM | Deskripsi Tugas |
|------|-----|-----------------|
| Nicholas Andhika Lucas | 13523014 | Membuat implementasi EDA, Membuat implementasi Split Training Set and Validation Set, Membuat implementasi Data Cleaning and Processing, Membuat implementasi Data Preprocessing: Labeling and Encoding, Membuat laporan: struktur, penjelasan tahap data cleaning dan data preprocessing  |
| Shanice Feodora Tjahjono | 13523097 | Membuat implementasi Data Preprocessing: Feature Scaling, Data, Normalization, Dimensionality Reduction, dan Pipeline, Menguji dan memperbaiki tahap Data Cleaning, Membuat laporan: penjelasan tahap Data Preprocessing |
| Jonathan Kenan Budianto | 13523139 | Implementasi model Decision Tree Learning, Membuat laporan: Penjelasan Implementasi Decision Tree Learning, Penjelasan hasil uji coba Decision Tree Learning, Uji coba model Decision Tree Learning |
| Mahesa Fadhillah Andre | 13523140 | Implementasi model Logistic Regression, Membuat laporan: Penjelasan implementasi Logistic Regression, Penjelasan hasil uji coba Logistic Regression, Uji coba model Logistic Regression |
| Muhammad Farrel Wibowo | 13523153 | Implementasi model SVM, Membuat laporan: Penjelasan implementasi SVM,  Penjelasan hasil uji coba SVM, Uji coba model SVM |
