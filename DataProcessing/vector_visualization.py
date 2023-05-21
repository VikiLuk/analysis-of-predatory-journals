import pandas as pd
import numpy as np
import openpyxl
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# VECTORIZATION --------------------------------------------------------------
# Load the input data from an Excel file
file_name1 = 'PredatoryJournalsMerged'
data1 = pd.read_excel(file_name1 + '.xlsx')
file_name2 = 'PatternRecognition'
data2 = pd.read_excel(file_name2 + '.xlsx')


# Define the tokenization function
def tokenize(text):
    # Convert the input text to a string
    text = str(text)

    # Check if the input text is empty or contains only whitespace
    if not text.strip():
        return []

    # Tokenize the text using NLTK
    tokens = gensim.utils.simple_preprocess(text)

    return tokens


# Tokenize the Abstract column
data1['abstract_tokens'] = data1['Abstract'].apply(tokenize)
data2['abstract_tokens'] = data2['Abstract'].apply(tokenize)

# Remove rows with empty token lists
data1 = data1[data1['abstract_tokens'].map(len) > 0]
data2 = data2[data2['abstract_tokens'].map(len) > 0]

# Tag the abstracts
tagged_abstracts1 = [TaggedDocument(doc, [i]) for i, doc in enumerate(data1['abstract_tokens'])]
tagged_abstracts2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(data2['abstract_tokens'])]

# Train the doc2vec model
model1 = Doc2Vec(tagged_abstracts1, vector_size=100, window=5, min_count=5, epochs=20)
model2 = Doc2Vec(tagged_abstracts2, vector_size=100, window=5, min_count=5, epochs=20)

# Get the vectors for the abstracts
vectors1 = [model1.infer_vector(doc.words) for doc in tagged_abstracts1]
vectors2 = [model2.infer_vector(doc.words) for doc in tagged_abstracts2]

# Add the vectors to the dataframe
data1['vector'] = vectors1
data2['vector'] = vectors2

# Save the data to a new file
output_data1 = data1[['abstract_tokens', 'vector']]
output_data2 = data2[['abstract_tokens', 'vector']]

output_data1.to_excel(file_name1 + '_vectorized.xlsx', index=False)
output_data2.to_excel(file_name2 + '_vectorized.xlsx', index=False)

# VISUALIZATION --------------------------------------------------------------
# Load the data from Excel file
df1 = pd.read_excel(file_name1 + '_vectorized.xlsx')
df2 = pd.read_excel(file_name2 + '_vectorized.xlsx')

# Convert the 'Vector' column from string to array
df1['vector'] = df1['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
df2['vector'] = df2['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Extract the feature vectors and the labels from the DataFrame
X1 = np.array(df1['vector'].tolist())
X2 = np.array(df2['vector'].tolist())

# Apply PCA with two principal components
pca = PCA(n_components=2)
X1_pca = pca.fit_transform(X1)
X2_pca = pca.fit_transform(X2)

# Calculate the centroid
centroid1 = np.mean(X1_pca, axis=0)
centroid2 = np.mean(X2_pca, axis=0)

# Calculate the distance between the centroids
centroid_distance = np.linalg.norm(centroid1 - centroid2)

# Print the distance between the centroids
print("Distance between centroids: ", centroid_distance)

# Plot the PCA results
fig, ax = plt.subplots()
ax.scatter(X1_pca[:, 0], X1_pca[:, 1], color='blue')
ax.scatter(X2_pca[:, 0], X2_pca[:, 1], color='green')

# Add centroid to the plot
ax.scatter(centroid1[0], centroid1[1], marker='+', s=200, linewidth=3, color='red')
ax.scatter(centroid2[0], centroid2[1], marker='x', s=200, linewidth=3, color='red')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Visualization with Centroid')
plt.show()


# SIMILARITY CALCULATION -----------------------------------------------------
cosine_sim = cosine_similarity(X1, X2)

# Calculate the percentage similarity
similarity_percentage = (cosine_sim * 100).round().astype(int)

# Count the number of rows with similarity in different ranges
count_similar = np.zeros(5, dtype=int)
for percentage in similarity_percentage.flatten():
    if 0 < percentage <= 20:
        count_similar[0] += 1
    elif 20 < percentage <= 40:
        count_similar[1] += 1
    elif 40 < percentage <= 60:
        count_similar[2] += 1
    elif 60 < percentage <= 80:
        count_similar[3] += 1
    elif percentage > 80:
        count_similar[4] += 1

count_similar = count_similar[::-1]

# Create the table
table_data = {
    'Similarity Range': ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'],
    'Data Points': count_similar
}

table = pd.DataFrame(table_data)

print(table)
