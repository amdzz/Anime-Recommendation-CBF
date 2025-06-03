# Laporan Proyek Machine Learning - Ahmad Zul Zhafran

## Project Overview

Anime telah menjadi bagian penting dari industri hiburan global, dengan jutaan penonton dari berbagai usia dan latar belakang di seluruh dunia. Dengan meningkatnya jumlah anime yang dirilis setiap tahunnya, pengguna menghadapi tantangan dalam menemukan tontonan baru yang sesuai dengan preferensi mereka. Dalam kondisi seperti ini, sistem rekomendasi menjadi sangat krusial untuk membantu pengguna menavigasi pilihan yang tersedia dan menemukan konten yang relevan dengan minat mereka.

Proyek ini bertujuan untuk membangun sistem rekomendasi anime berbasis konten (Content-Based Filtering) yang merekomendasikan anime lain berdasarkan genre dari anime yang telah disukai pengguna sebelumnya. Pendekatan ini memungkinkan sistem untuk menyarankan tontonan yang serupa dalam hal konten, tanpa bergantung pada interaksi pengguna lain. Masalah yang diselesaikan dalam proyek ini adalah kebingungan pengguna dalam memilih anime yang sesuai dengan selera mereka, terutama bagi pengguna baru yang belum banyak menonton anime. Sistem rekomendasi yang dibangun mampu mengurangi waktu pencarian dan meningkatkan pengalaman pengguna dalam menjelajahi konten anime. Putri H. D. et al. (2023) telah mengaplikasikan teknik Collaborative Filtering dan Content-Based Filtering untuk sistem rekomendasi anime. Hasilnya menunjukkan bahwa meskipun CF efektif dalam membuat rekomendasi berdasarkan preferensi serupa pengguna, CBF tampaknya memberikan kepuasan yang lebih tinggi dengan berfokus pada karakteristik konten animasi untuk membuat rekomendasi yang dipersonalisasi. 

Putri, H. D. dan Faisal, M.  (2023). Analyzing the effectiveness of collaborative filtering and content-based filtering methods in anime recommendation systems. Jurnal Komtika (Komputasi dan Informatika), 7 (2). pp. 124-133. ISSN 2580734X


## Business Understanding

### Problem Statements

- Pengguna kesulitan menemukan rekomendasi anime yang sesuai dengan preferensi genre mereka tanpa harus menelusuri banyak judul secara manual.
- Tidak semua pengguna memiliki riwayat interaksi atau penilaian terhadap anime yang bisa dimanfaatkan untuk sistem rekomendasi berbasis user.
- Metode rekomendasi berbasis kesamaan konten, khususnya genre, belum menjadi fokus utama dalam sistem rekomendasi anime.

### Goals

- Membangun model sistem rekomendasi yang dapat menyarankan anime dengan genre serupa dari anime yang diberikan.
- Menerapkan pendekatan Content-Based Filtering agar sistem dapat tetap memberikan rekomendasi meskipun tanpa data riwayat pengguna.
- Membantu pengguna menemukan tontonan baru yang relevan dengan preferensi genre mereka secara otomatis.

## Data Understanding

Dataset yang digunakan dalam proyek ini terdiri dari data anime yang diambil dari file animes.csv yang berasal dari [Kaggle](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews). Dataset ini berisi informasi tentang anime-anime dari website MyAnimelist beserta atribut pendukungnya. Dataset memiliki total 19311 data anime dengan berbagai fitur penting seperti judul, genre, durasi, skor, dan lain-lain.

Variabel-variabel utama yang terdapat dalam dataset animes.csv adalah sebagai berikut:
- uid : ID unik untuk setiap anime.
- title : Judul utama anime.
- synopsis : Ringkasan cerita dari anime.
- genre : Kategori atau jenis genre anime, misalnya Action, Comedy, Fantasy, dll.
- aired : Periode tanggal anime ditayangkan.
- episodes : Jumlah episode yang tersedia pada anime.
- members : Jumlah anggota komunitas yang menonton atau mengikuti anime.
- popularity : Peringkat popularitas anime berdasarkan jumlah pengikut.
- ranked : Peringkat anime secara keseluruhan berdasarkan penilaian pengguna.
- score : Skor rata-rata penilaian dari pengguna.
- img_url : URL gambar poster anime.
- link : Tautan menuju halaman detail anime.

Variabel-variabel ini akan digunakan sebagai dasar dalam membangun sistem rekomendasi berbasis konten dengan fokus utama pada fitur genre.

![image](https://github.com/user-attachments/assets/bae638a3-ac1f-4b67-b66c-1b9fe2765b8f)

DataFrame yang ditampilkan memiliki 19.311 entri dengan indeks dari 8 hingga 19.318 dan terdiri dari 11 kolom, yaitu id, title, synopsis, genre, aired, episodes, members, popularity, ranked, score, img_url, dan link. Semua kolom memiliki data non-null sebanyak 19.311, kecuali synopsis (18.336), episodes (18.605), ranked (16.099), dan score (18.732). yang menunjukkan adanya missing values yakni: 
- uid : 0
- title : 0
- synopsis : 975
- genre : 0
- aired : 0
- episodes : 706
- members : 0
- popularity : 0
- ranked : 3.212
- score : 579
- img_url : 180
- link : 0

Tipe data terdiri dari float64 (3 kolom: episodes, ranked, score), int64 (3 kolom: id, members, popularity), dan object (6 kolom: title, synopsis, genre, aired, img_url, link).

![image](https://github.com/user-attachments/assets/3e569f30-4e26-4aec-bd76-dc5979c678e1)
![image](https://github.com/user-attachments/assets/7a338e0e-0714-4918-9bef-641569fdbf13)

Grafik di atas adalah visualisasi dari "Top 10 Genre Anime Terpopuler" yang menunjukkan distribusi jumlah anime berdasarkan genre. Genre "Comedy" menjadi yang terpopuler dengan jumlah anime terbanyak, dengan 6461 judul, diikuti oleh "Action" dengan 4215 judul. Genre "Fantasy" 3466 judul, kemudian "Adventure" memiliki 3143 judul. Sedangkan "Drama" memiliki 3004 judul dan "Sci-Fi" berada di 2832 judul. Genre "Hentai" memiliki 2574 judul, "Kids" 2549 judul, dan "Shounen" memiliki 2322 judul, dengan "Romance" menempati posisi terbawah di daftar ini dengan 2152 judul. Grafik ini menunjukkan bahwa "Comedy" dan "Action" adalah genre yang paling dominan dalam dataset anime tersebut, sementara "Romance" memiliki popularitas yang lebih rendah dibandingkan genre lainnya dalam 10 besar.

![image](https://github.com/user-attachments/assets/63411486-3855-4481-b68d-855504221f93)

Output statistik deskriptif dari DataFrame untuk kolom score, ranked, popularity, dan episodes memberikan gambaran distribusi data dalam dataset anime tersebut. Secara keseluruhan, data ini mencerminkan variasi yang signifikan dalam skor, peringkat, popularitas, dan jumlah episode anime dalam dataset.

## Data Preparation
Tahap data preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pelatihan model sudah bersih, valid, dan relevan. Proses ini dimulai dengan pemilihan fitur-fitur yang paling berpengaruh terhadap model sistem rekomendasi CBF, yaitu uid, title, dan genre. Pemilihan ini bertujuan untuk menyederhanakan data dan hanya mempertahankan informasi yang diperlukan dalam proses pemodelan. Setelah itu kolom uid diganti namanya menjadi anime_id untuk memperjelas. Kemudian diterapkan beberapa teknik dalam tahap data preparation yakni:

### 1. Penanganan Missing Values

![image](https://github.com/user-attachments/assets/a78ae297-2f2a-4f88-86ce-4564f35e6075)

Output dari perintah df.isnull().sum() menunjukkan jumlah missing values pada kolom anime_id, title, dan genre dalam DataFrame. Ketiga kolom tersebut, yaitu anime_id, title, dan genre, masing-masing memiliki 0 entri yang bernilai null, yang berarti tidak ada data yang hilang pada kolom-kolom ini.

### 2. Pengurutan Berdasarkan ID

Data kemudian diurutkan berdasarkan kolom anime_id untuk memperjelas identitas setiap entri anime. Pengurutan ini dilakukan untuk menjaga konsistensi saat proses evaluasi atau pelacakan hasil rekomendasi nantinya. Data yang telah di urutkan kemudian dimasukkan ke fix_df.

![image](https://github.com/user-attachments/assets/7cdddc88-0bb1-4c2c-b5dc-b0eee53df2ec)

### 3. Pembersihan dan Penyederhanaan Genre

Kolom genre awalnya berisi nilai berupa string dari list genre, misalnya "[‘Action’, ‘Comedy’]". Format ini tidak ideal untuk pemrosesan teks. Oleh karena itu, dilakukan parsing menggunakan ast.literal_eval() untuk mengonversi string menjadi list Python. Setelah itu, genre dibersihkan dari karakter yang tidak perlu, dan hanya satu genre utama yang diambil dari setiap entri untuk menyederhanakan proses rekomendasi yakni hanya mengambil genre pertama dari list genre.

![image](https://github.com/user-attachments/assets/e76174c0-5cb5-4147-bd02-e6f7e570838c)

![image](https://github.com/user-attachments/assets/db40865d-8630-44da-b21e-a6ede934f6f4)

Setelah menyederhanakan genre, muncul sejumlah missing values pada kolom tersebut. Missing values ini akan ditindak dengan cara dihapus.

### 4. Penyesuaian Format Genre

Pada langkah ini, dilakukan penggantian karakter spasi pada nilai di kolom genre menjadi garis bawah (_). Langkah ini penting agar genre yang terdiri dari dua kata atau lebih, seperti "Slice of Life" atau "Sci fi", tidak terpotong saat diproses oleh algoritma TF-IDF yang secara default memisahkan kata berdasarkan spasi. Dengan mengubah format menjadi "Slice_of_Life", kita memastikan bahwa genre tersebut diperlakukan sebagai satu entitas utuh, sehingga representasi fitur menjadi lebih akurat dan tidak kehilangan makna genre aslinya.

### 5. Penghapusan Duplikasi

Data duplikat yang memiliki informasi identik lebih dari satu kali dihapus untuk memastikan bahwa model tidak belajar dari data yang sama secara berulang. Hal ini juga dilakukan agar hasil analisis menjadi lebih valid dan tidak bias.

![image](https://github.com/user-attachments/assets/7712ad28-34c2-4790-b385-8229db0aad88)

Output dari kode tersebut menunjukkan proses pembersihan data duplikat pada DataFrame fix_df. Pada langkah pertama, perintah fix_df.duplicated().sum() menghasilkan nilai 3.088, yang berarti terdapat 3.088 baris duplikat ditemukan. Data duplikat ini kemudian dihapus dan menyisakan 16141 entri data pada dataset.

### 6. Penyusunan Dataset Ringkas untuk Modelling

Sebelum masuk ke tahap pemodelan, dilakukan penyusunan ulang dataset dengan mengambil hanya kolom-kolom yang relevan, yaitu anime_id, title, dan genre. Masing-masing kolom kemudian dikonversi ke dalam bentuk list untuk memudahkan pembentukan struktur data baru yang lebih sederhana. Selanjutnya, dibuat sebuah DataFrame baru bernama anime_new yang berisi tiga kolom: id, anime_title, dan anime_genre. Dataset ini disiapkan khusus sebagai input utama pada tahap content-based filtering, dengan fokus utama pada kesesuaian antara judul dan genre anime. Langkah ini bertujuan untuk menyederhanakan data dan memfokuskan informasi hanya pada elemen-elemen yang diperlukan dalam proses sistem rekomendasi berbasis konten.

### 7. Inisialisasi TF-IDF Vectorizer
TfidfVectorizer digunakan untuk mengubah data anime_genre menjadi vektor numerik berdasarkan frekuensi kemunculan genre dalam setiap entri anime. Parameter token_pattern=r"(?u)\b[\w\-]+\b" digunakan untuk mempertahankan genre yang mengandung tanda hubung atau spasi yang telah diubah sebelumnya menjadi underscore (contoh: "slice_of_life").

### 8. Pembuatan TF-IDF Matrix
Data anime_genre diubah menjadi matriks TF-IDF, di mana setiap baris merepresentasikan sebuah anime, dan setiap kolom merepresentasikan sebuah genre. Nilai dalam matriks menunjukkan bobot genre tersebut pada masing-masing anime.


## Modeling

Tahapan ini merupakan inti dari pembangunan sistem rekomendasi berbasis konten (content-based filtering), yang bertujuan untuk menyarankan anime berdasarkan kemiripan konten, khususnya dari sisi genre. Model yang digunakan dalam proyek ini adalah Content-Based Recommendation dengan pendekatan Text Feature Extraction menggunakan TF-IDF. Content-based filtering adalah metode yang digunakan dalam sistem rekomendasi dan analisis data yang berfokus pada karakteristik atau konten dari item-item yang ingin direkomendasikan atau dianalisis. Pendekatan ini menggunakan atribut-atribut atau fitur-fitur item untuk menentukan kesamaan antara item yang ada dan preferensi pengguna (Maulid, 2023).

Setelah setiap anime direpresentasikan sebagai vektor TF-IDF, Cosine Similarity digunakan untuk menghitung kemiripan antar-anime berdasarkan genre. Dalam sebuah proses menghitung cosine similarity, pertama yang dilakukannya adalah melakukan sebuah perkalian skalar antara query dengan sebuah dokumen yang kemudian ditambahkan, lalu itu melakukan perkalian antara ukuran panjang dari dokumen dengan ukuran panjang query yang telah dikuadratkan, kesamaan kosinus mengukur kesamaan antara dua vektor ruang hasil kali dalam dengan cosinus sudut antara dua vektor dan menentukan apakah dua vektor menunjuk ke arah yang kira-kira sama (Sholihin, 2022). Cosine Similarity memberikan nilai antara 0 (tidak mirip) hingga 1 (identik). Nilai kemiripan ini memungkinkan sistem untuk mengurutkan dan memilih anime yang paling mirip dengan judul input.

### Perhitungan Cosine Similarity
Matriks kesamaan dihitung menggunakan cosine similarity terhadap TF-IDF matrix. Cosine similarity mengukur sejauh mana dua anime memiliki kemiripan berdasarkan distribusi bobot genre mereka.

### Pembuatan DataFrame Cosine Similarity
Hasil dari cosine similarity disimpan dalam sebuah DataFrame cosine_sim_df, dengan baris dan kolom berisi judul anime. Setiap nilai menunjukkan tingkat kesamaan antara dua anime berdasarkan kontennya.

![image](https://github.com/user-attachments/assets/6f1e88a3-d7ea-40ad-a82f-49ad917071e0)

### Output Top-N 

Berikut merupakan output Top-10 hasil rekomendasi untuk anime 'Cowboy Bebop' :

![image](https://github.com/user-attachments/assets/c241d9e3-b5f5-4d9d-80f6-05e08e0e73c4)

### Daftar Pustaka

Maulid, Reyvan. (2023). Content Based Filtering dalam Algoritma Data Science. Diakses pada 01 Juni 2025 dari https://dqlab.id/content-based-filtering-dalam-algoritma-data-science.

Sholihin, Ahmad. (2022). Rekomendasi menggunakan Content Based Filtering dan Collaborative Filtering. Diakses pada 01 Juni 2025 dari https://medium.com/@ahmadsholihin1705/rekomendasi-menggunakan-content-based-filtering-dan-collaborative-filtering-2aab2aec8ebe.

## Evaluation
Untuk mengevaluasi performa sistem rekomendasi berbasis konten ini, digunakan metrik Precision. Precision adalah salah satu metrik evaluasi yang umum digunakan dalam sistem informasi untuk mengukur ketepatan dari hasil rekomendasi. Metrik ini menunjukkan seberapa banyak rekomendasi yang diberikan benar-benar relevan dengan item input. Berikut merupakan rumus precision:

![image](https://github.com/user-attachments/assets/ff2bb892-912b-431b-9c29-5688c87bb393)

Pada anime 'Cowboy Bebop' sebagai input, sistem menghasilkan 10 rekomendasi teratas berdasarkan kemiripan genre. Dari 10 rekomendasi tersebut, sebanyak 10 memiliki genre yang sama dengan Cowboy Bebop, yaitu Action. Maka nilai precision-nya adalah:

![image](https://github.com/user-attachments/assets/c4e82865-0dae-45f5-af20-96f67b9ef174)

![image](https://github.com/user-attachments/assets/74223a48-bcd1-4706-9653-6667cbf3d10a)

### Keterkaitan Evaluasi Model dengan Business Understanding

#### Problem Statement
Model sistem rekomendasi berbasis content-based filtering yang dibangun telah berhasil menjawab seluruh pernyataan masalah yang telah dirumuskan:
- Model mampu menyarankan judul anime berdasarkan kemiripan genre dari anime yang diberikan. Hal ini memudahkan pengguna dalam menemukan tontonan baru tanpa harus melakukan pencarian manual, menjawab permasalahan kesulitan dalam eksplorasi konten.
- Karena pendekatan yang digunakan tidak memerlukan data interaksi pengguna, sistem tetap dapat berfungsi optimal meskipun pengguna belum memiliki riwayat rating atau aktivitas menonton. Ini menjawab kebutuhan akan sistem rekomendasi yang inklusif bagi pengguna baru.
- Evaluasi menggunakan metrik precision menunjukkan bahwa sebagian besar rekomendasi yang diberikan relevan dengan genre anime input. Nilai precision mencapai angka tinggi yakni 1.0 yang menunjukkan bahwa sistem sudah mampu untuk mengidentifikasi kesamaan genre secara fungsional.

#### Goals
Dengan demikian, seluruh tujuan dari tahap Business Understanding dapat dikatakan tercapai:
- Sistem rekomendasi berhasil dibangun dan mampu memberikan saran anime berdasarkan genre.
- Pendekatan content-based filtering telah berhasil diterapkan dan berjalan baik tanpa memerlukan data riwayat pengguna.
- Pengguna dapat dengan lebih mudah menemukan judul-judul anime yang relevan dengan preferensi mereka secara otomatis, sehingga pengalaman eksplorasi konten menjadi lebih efisien dan personal.

