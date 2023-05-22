import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt

# Load datasets
predatory_df = pd.read_excel('PredatoryJournalsMerged.xlsx')
non_predatory_df = pd.read_excel('PatternRecognition.xlsx')


# Preprocess abstracts
def preprocess_abstracts(df):
    return df['Abstract'].apply(lambda x: x.lower().split() if isinstance(x, str) else [])


predatory_abstracts = preprocess_abstracts(predatory_df)
non_predatory_abstracts = preprocess_abstracts(non_predatory_df)

# Train Word2Vec model
model = Word2Vec(sentences=pd.concat([predatory_abstracts, non_predatory_abstracts]), vector_size=100, window=5,
                 min_count=1, workers=4)


# Vectorize abstracts
def vectorize_abstracts(abstracts):
    max_length = max(len(abstract) for abstract in abstracts)
    vectors = np.zeros((len(abstracts), max_length, 100))
    for i, abstract in enumerate(abstracts):
        for j, word in enumerate(abstract):
            vectors[i, j, :] = model.wv[word]
    return np.mean(vectors, axis=1)


predatory_vectors = vectorize_abstracts(predatory_abstracts)
non_predatory_vectors = vectorize_abstracts(non_predatory_abstracts)


# Visualize using PCA
def visualize_pca(vectors, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)
    plt.figure(figsize=(10, 10))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='coolwarm')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Visualization')
    plt.colorbar()

    # Add centroid to plot
    centroid = np.mean(pca_result[labels == 0], axis=0)
    plt.scatter(centroid[0], centroid[1], marker='x', s=200, linewidths=3, color='black')

    plt.show()


combined_vectors = np.vstack((predatory_vectors, non_predatory_vectors))
labels = np.concatenate((np.ones(len(predatory_vectors)), np.zeros(len(non_predatory_vectors))))
visualize_pca(combined_vectors, labels)

# Calculate centroid of non-predatory dataset
centroid = np.mean(non_predatory_vectors, axis=0)

# Calculate cosine similarity
predatory_similarity = cosine_similarity(predatory_vectors, centroid.reshape(1, -1))
non_predatory_similarity = cosine_similarity(non_predatory_vectors, centroid.reshape(1, -1))


# Create similarity table
def create_similarity_table(similarity_scores):
    ranges = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    similarity_percentages = similarity_scores * 100
    counts = Counter([tuple(r) for r in ranges for s in similarity_percentages if r[0] <= s <= r[1]])
    return pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index().rename(
        columns={'index': 'Range'})


# similarity_table = create_similarity_table(np.concatenate((predatory_similarity, non_predatory_similarity)))
similarity_table = create_similarity_table(predatory_similarity)
print(similarity_table)
