import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# ==============================================================================
# 1. MEMUAT DATASET (DATA LOADING)
# ==============================================================================
df = pd.read_csv('spotify_songs.csv')
print(f"Jumlah awal baris & kolom: {df.shape}")

# ==============================================================================
# 2. DATA PREPROCESSING & FEATURE SELECTION
# ==============================================================================
features = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

df_clean = df.dropna(subset=features).drop_duplicates(subset=['track_name', 'track_artist']).copy()

if len(df_clean) > 5000:
    df_clean = df_clean.sample(n=5000, random_state=42).reset_index(drop=True)

X = df_clean[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 3. MENCARI JUMLAH KLASTER OPTIMAL (ELBOW METHOD & SILHOUETTE SCORE)
# ==============================================================================
inertia = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(K_range, inertia, marker='o', linestyle='--', color='b')
ax[0].set_title('Elbow Method (Inertia vs Jumlah Klaster)')
ax[0].set_xlabel('Jumlah Klaster (k)')
ax[0].set_ylabel('Inertia (WCSS)')

ax[1].plot(K_range, sil_scores, marker='s', linestyle='-', color='g')
ax[1].set_title('Silhouette Score vs Jumlah Klaster')
ax[1].set_xlabel('Jumlah Klaster (k)')
ax[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig('evaluasi_klaster.png', dpi=300)
plt.show()

# ==============================================================================
# 4. PEMODELAN K-MEANS & DETEKSI GENRE DOMINAN PER KLASTER
# ==============================================================================
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_clean['Cluster_ID'] = kmeans_final.fit_predict(X_scaled)

# Karakteristik Mood berdasarkan profil audio
mood_descriptions = {
    0: 'High-Energy Instrumental / Electronic Beats',
    1: 'Fast, Loud & Intense (Workout / Rock Anthem)',
    2: 'Upbeat, Groovy & Cheerful (Dance / Party)',
    3: 'Chill, Acoustic & Mellow (Relaxing)'
}

# Otomatis mendeteksi 2 genre asli terbanyak (playlist_genre) pada setiap klaster
cluster_full_labels = {}
for cid in range(optimal_k):
    subset = df_clean[df_clean['Cluster_ID'] == cid]
    top_genres = subset['playlist_genre'].value_counts().head(2).index.str.upper().tolist()
    genre_str = " & ".join(top_genres)
    cluster_full_labels[cid] = f"Klaster {cid} - {mood_descriptions[cid]} [Dominan: {genre_str}]"

df_clean['Label_Klaster'] = df_clean['Cluster_ID'].map(cluster_full_labels)

final_sil = silhouette_score(X_scaled, df_clean['Cluster_ID'])
print(f"\nSilhouette Score akhir untuk k={optimal_k}: {final_sil:.4f}")

# ==============================================================================
# 5. VISUALISASI HASIL KLASTER DENGAN PCA (2D)
# ==============================================================================
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df_clean['PCA1'] = pca_components[:, 0]
df_clean['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df_clean, x='PCA1', y='PCA2', 
    hue='Label_Klaster', palette='viridis', alpha=0.6, s=40
)
plt.title(f'Visualisasi Klaster Lagu Spotify & Genre Dominannya (k={optimal_k})')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
plt.legend(title='Profil Klaster & Genre Terkait', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('visualisasi_pca_klaster.png', dpi=300)
plt.show()

# ==============================================================================
# 6. OUTPUT LENGKAP: DISTRIBUSI GENRE, RATA-RATA FITUR, & CONTOH LAGU
# ==============================================================================
print("\n=== DISTRIBUSI GENRE ASLI DI DALAM SETIAP KLASTER (%) ===")
genre_dist = pd.crosstab(df_clean['Label_Klaster'], df_clean['playlist_genre'], normalize='index') * 100
print(genre_dist.round(1))
genre_dist.round(1).to_csv('distribusi_genre_per_klaster.csv')

print("\n=== RATA-RATA FITUR AUDIO PER KLASTER ===")
cluster_profile = df_clean.groupby('Cluster_ID')[features].mean()
print(cluster_profile.round(3).T)
cluster_profile.round(3).T.to_csv('profil_klaster_spotify.csv')

print("\n=== DETAIL KLASTER, GENRE TERKAIT, & CONTOH LAGU ===")
for cid in range(optimal_k):
    print(f"\n==================================================================")
    print(f"{cluster_full_labels[cid]}")
    print(f"==================================================================")
    
    # Menampilkan rincian proporsi genre pada klaster ini
    genre_counts = df_clean[df_clean['Cluster_ID'] == cid]['playlist_genre'].value_counts(normalize=True) * 100
    rincian_genre = ", ".join([f"{g.upper()} ({p:.1f}%)" for g, p in genre_counts.head(3).items()])
    print(f"Komposisi Genre Teratas : {rincian_genre}\n")
    
    # Menampilkan 3 contoh lagu
    sample_songs = df_clean[df_clean['Cluster_ID'] == cid][['track_name', 'track_artist', 'playlist_genre']].head(3)
    print(sample_songs.to_string(index=False))