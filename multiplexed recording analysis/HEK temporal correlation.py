from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
import umap
from matplotlib.colors import LinearSegmentedColormap
import csv
from itertools import zip_longest


## =============================Functions==========================================

def time_lagged_correlation(signal1, signal2, max_lag):
    cross_corr = correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full')
    cross_corr /= (np.std(signal1) * np.std(signal2) * len(signal1))
    lags = np.arange(-len(signal1) + 1, len(signal1))
    mid = len(cross_corr) // 2
    relevant_lags = lags[mid - max_lag: mid + max_lag + 1]
    relevant_corrs = cross_corr[mid - max_lag: mid + max_lag + 1]
    return relevant_lags, relevant_corrs


## =============================Script==========================================

# Load data
df_cre = pd.read_csv("Cre data.csv")
df_fram = pd.read_csv("FRAM data.csv")

# Compute time-lagged correlations
max_lag = 100
time_lag_corr_matrix = []
day_of_stimuli = int((2.25/4) * 1000 +0.5)
for i in range(1, 78):
    data_cre = df_cre[df_cre.columns[i]].tolist()
    data_fram = df_fram[df_fram.columns[i]].tolist()

    _, correlations = time_lagged_correlation(data_cre[day_of_stimuli:750], data_fram[day_of_stimuli:750], max_lag) 
    time_lag_corr_matrix.append(correlations)

signals = np.array(time_lag_corr_matrix)


pca_model = PCA(n_components = 10)  
pca_result = pca_model.fit_transform(signals)

## UMAP on PCA result
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.3, 
    metric='cosine',
    random_state=12 
)
umap_result = umap_model.fit_transform(pca_result)

# KMeans on UMAP result
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(umap_result)

# Plotting
df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
df['Cluster'] = clusters

plt.figure(figsize=(6, 4.5))
sns.scatterplot(data=df, x='UMAP1', y='UMAP2', 
                hue='Cluster', palette=['#BEBEBE', '#000000'], alpha=0.8)
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.legend(title='Cluster')
plt.legend([],[], frameon=False) 
sns.despine()
plt.grid(False)
plt.show()


## Plot heatmap for visualize these two clusters
unique_clusters = np.unique(clusters)
for cluster_id in unique_clusters:
    cluster_1 = df[df['Cluster'] == cluster_id]

    heatmap_data = []
    for i in cluster_1.index:
        heatmap_data.append(time_lag_corr_matrix[i])
    print(len(heatmap_data))

    # heatmap
    colors = ['#9bbdde', '#FFFFFF', '#ce6b6b']
    custom_cmap = LinearSegmentedColormap.from_list("custom_rwb", colors)

    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap_data, aspect="auto", cmap=custom_cmap, vmin=0, vmax=1,
           extent=[-100, 100, 0, len(heatmap_data[0])])

    plt.colorbar(label="Normalized Red Signal Intensity") 
    plt.xlabel("Lag (Timepoints)")
    plt.ylabel("Sample Index")
    plt.title(f'Cluster {cluster_id}')
    plt.show()


## Calculate baseline fluctuation for the two clusters
num1 = int((1.5/4) * 1000 +0.5) # baseline starts from day 1.5
print(num1)

num2 = int((2.25/4) * 1000 +0.5) # baseline ends at day 2.25
print(num2)

fluctuations_1 = []
fluctuations_0 = []

cluster_members = df[df['Cluster'] == 0].index  
for j in cluster_members:
    data_cre = df_cre[df_cre.columns[j+1]].tolist()
    data_fram = df_fram[df_fram.columns[j+1]].tolist()
    data_fluctuation = data_fram[num1:num2]
    result = np.sum(np.abs(np.diff(data_fluctuation)))
    fluctuations_0.append(result)

cluster_members = df[df['Cluster'] == 1].index  
for i in cluster_members:
    data_fram = df_fram[df_fram.columns[i+1]].tolist()
    data_cre = df_cre[df_cre.columns[i+1]].tolist()   
    data_fluctuation = data_fram[num1:num2]
    result = np.sum(np.abs(np.diff(data_fluctuation)))
    fluctuations_1.append(result)


# save data to csv file
with open('fluctuations data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['cluster 0', 'cluster 1'])  
    for row in zip_longest(fluctuations_0, fluctuations_1, fillvalue=''):
        writer.writerow(row)


