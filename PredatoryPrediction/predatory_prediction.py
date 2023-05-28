import os
import shutil
from collections import Counter

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tensorflow import keras

absl.logging.set_verbosity(absl.logging.ERROR)


def main():
    print("Please wait a moment . . .")

    # Load training datasets
    try:
        predatory_df = read_dataset_dir('datasets/training/predatory')
        non_predatory_df = read_dataset_dir('datasets/training/non-predatory')
    except ValueError:
        print("Error: Invalid or Empty training data.")
        return

    # Preprocess abstracts of training datasets
    predatory_abstracts = preprocess_abstracts(predatory_df)
    non_predatory_abstracts = preprocess_abstracts(non_predatory_df)

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=pd.concat([predatory_abstracts, non_predatory_abstracts]), vector_size=300,
                              window=5, min_count=1, workers=4)

    # Vectorize abstracts
    predatory_vectors = vectorize_abstracts(predatory_abstracts, word2vec_model)
    non_predatory_vectors = vectorize_abstracts(non_predatory_abstracts, word2vec_model)

    # Combine Vectors
    combined_vectors = np.vstack((predatory_vectors, non_predatory_vectors))

    # Create Labels
    labels = np.concatenate((np.ones(len(predatory_vectors)), np.zeros(len(non_predatory_vectors))))

    # Program Cycle
    while True:
        os.system('cls')
        print("###############################################################")
        print("#                    PREDATORY PREDICTION                     #")
        print("###############################################################")
        print("# (p) PCA Visualization and Similarity Table of training data #")
        print("# (c) Create and Train model                                  #")
        print("# (l) Load Model and Predict                                  #")
        print("# (e) Exit                                                    #")
        print("###############################################################")

        menu_opt = input("<< ").lower()
        match menu_opt:
            case 'p':
                # PCA Visualization and Similarity Table of training data
                visualize_data(non_predatory_vectors, predatory_vectors, combined_vectors, labels)
            case 'c':
                # Create new model and train
                create_and_train(combined_vectors, labels, word2vec_model)
            case 'l':
                # Load model and predict
                load_and_predict()
            case 'e':
                print("Bye!")
                break


def visualize_data(non_predatory_vectors, predatory_vectors, combined_vectors, labels):
    os.system('cls')
    # Calculate centroid of non-predatory dataset
    centroid = np.mean(non_predatory_vectors, axis=0)

    # Calculate cosine similarity
    predatory_similarity = cosine_similarity(predatory_vectors, centroid.reshape(1, -1))
    non_predatory_similarity = cosine_similarity(non_predatory_vectors, centroid.reshape(1, -1))

    # Create similarity table
    similarity_table = create_similarity_table(predatory_similarity)
    print("Similarity table: ")
    print(similarity_table)

    # Visualize vectors
    visualize_pca(combined_vectors, labels)

    input("\nPress Enter to continue . . . ")


def create_and_train(combined_vectors, labels, word2vec_model):
    os.system('cls')

    # Create the neural network
    input_shape = combined_vectors.shape[1:]
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Split training and test data
    X_train, X_test, y_train, y_test = train_test_split(combined_vectors, labels, test_size=0.2, random_state=42,
                                                        shuffle=True)

    print("Number of epochs: \n(d) default (75) \n(c) custom")
    while True:
        choice = input("<< ").lower()
        if choice == 'c':
            while True:
                epochs = input("epochs: ")
                try:
                    epochs = int(epochs)
                    if epochs < 1:
                        raise ValueError
                    break
                except ValueError:
                    print("Value should be a positive integer.")
                    continue
            break
        elif choice == 'd':
            epochs = 75
            break
    os.system('cls')

    # Train the neural network
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, epochs=epochs)

    # Predictions on Test data
    predictions = map(lambda x: np.round(x), model.predict(X_test))
    print("\nAccuracy score:", accuracy_score(y_test, np.array(list(predictions))))

    # Save the trained model and weights
    model.save('keras/model')
    model.save_weights('keras/model_weights')

    # Save Word2Vec model
    word2vec_model.save('word2vec/word2vec_model')

    input("\nPress Enter to continue . . . ")
    return model


def load_and_predict():
    os.system('cls')
    model = keras.models.load_model('keras/model')
    model.load_weights('keras/model_weights').expect_partial()

    # Use the neural network to make predictions
    new_dataset = read_dataset_dir('datasets/prediction')
    new_abstracts = preprocess_abstracts(new_dataset)

    # Train Word2Vec model
    word2vec_model = Word2Vec.load('word2vec/word2vec_model')
    word2vec_model.build_vocab(new_abstracts, update=True)

    new_vectors = vectorize_abstracts(new_abstracts, word2vec_model)

    predictions = model.predict(new_vectors)

    sum_predatory = 0
    sum_potentially = 0
    sum_non_predatory = 0
    sum_predictions = 0
    n = len(predictions)

    # Print the predictions
    for i, prediction in enumerate(predictions):
        pre = f"Abstract {i + 1} = {prediction[0] * 100} % "

        if 0 <= prediction <= 0.4:
            pre = pre + "(NP)"
            sum_non_predatory = sum_non_predatory + 1
        elif 0.4 < prediction < 0.6:
            pre = pre + "(PP)"
            sum_potentially = sum_potentially + 1
        elif 0.6 <= prediction <= 1:
            pre = pre + "(Pr)"
            sum_predatory = sum_predatory + 1

        sum_predictions = sum_predictions + prediction[0]
        print(pre)

    print("\nPr: predatory")
    print("PP: potentially predatory")
    print("NP: non-predatory")

    # Percentual
    predatory_perc = (sum_predatory / n) * 100
    potentially_perc = (sum_potentially / n) * 100
    non_predatory_perc = (sum_non_predatory / n) * 100
    prediction_perc = (sum_predictions / n) * 100

    print("\nPr:", sum_predatory, "/", n)
    print("PP:", sum_potentially, "/", n)
    print("NP:", sum_non_predatory, "/", n)

    print("\nPR =", predatory_perc, "%")
    print("PP =", potentially_perc, "%")
    print("NP =", non_predatory_perc, "%")

    print("\nPrediction =", prediction_perc, "%")

    # Assumption
    if 0 <= prediction_perc <= 40:
        assump = 'non-predatory'
    elif 40 < prediction_perc < 60:
        assump = 'potentially predatory'
    elif 60 <= prediction_perc <= 100:
        assump = 'predatory'

    print("\nProvided journal is", assump, "according to prediction.")
    print("Would you like to classify it as such and save it to the database? (y/n)")
    while True:
        choice = input("<< ").lower()
        if choice == 'y':
            save_to_database(assump)
            break
        elif choice == 'n':
            break

    input("\nPress Enter to continue . . . ")


# Load datasets
def read_dataset_dir(directory):
    dfs = []
    try:
        for file_name in os.listdir(directory):
            if file_name.endswith('.xlsx'):
                # Read the Excel file into a dataframe
                file_path = os.path.join(directory, file_name)
                try:
                    df = pd.read_excel(file_path)
                    # Append the dataframe to the list
                    dfs.append(df)
                except pd.errors.EmptyDataError:
                    print(f"EmptyDataError: Skipping empty file: {file_path}")
                except Exception as e:
                    print(f"Error reading file: {file_path}\n{type(e).__name__}: {str(e)}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    except PermissionError:
        print(f"Permission error: Unable to access files in directory: {directory}")
    except Exception as e:
        print(f"Error accessing directory: {directory}\n{type(e).__name__}: {str(e)}")

    return pd.concat(dfs, ignore_index=True)


# Preprocess abstracts
def preprocess_abstracts(df):
    return df['Abstract'].apply(lambda x: x.lower().split() if isinstance(x, str) else [])


# Vectorize abstracts
def vectorize_abstracts(abstracts, word2vec_model):
    max_length = max(len(abstract) for abstract in abstracts)
    vectors = np.zeros((len(abstracts), max_length, 300))
    for i, abstract in enumerate(abstracts):
        for j, word in enumerate(abstract):
            vectors[i, j, :] = word2vec_model.wv[word]
    return np.mean(vectors, axis=1)


# Visualize using PCA
def visualize_pca(vectors, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectors)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='coolwarm')
    plt.xlabel('Predatory Score')
    plt.ylabel('Non-Predatory Score')
    plt.title('PCA Visualization of Predatory and Non-Predatory Clusters')
    plt.colorbar(scatter)

    # Add centroid to plot
    centroid = np.mean(pca_result[labels == 0], axis=0)
    plt.scatter(centroid[0], centroid[1], marker='x', s=200, linewidths=3, color='black')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Non-Predatory', markerfacecolor=scatter.cmap(0.),
                   markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Predatory', markerfacecolor=scatter.cmap(1.), markersize=10)
    ]
    plt.legend(handles=legend_elements)

    plt.grid(True)
    plt.show()


# Create similarity table
def create_similarity_table(similarity_scores):
    ranges = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    similarity_percentages = similarity_scores * 100
    counts = Counter([tuple(r) for r in ranges for s in similarity_percentages if r[0] <= s <= r[1]])
    return pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index().rename(
        columns={'index': 'Range'})


def copy_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    files = os.listdir(source_dir)
    for file in files:
        source_file = os.path.join(source_dir, file)
        if os.path.isfile(source_file):
            destination_file = os.path.join(destination_dir, file)
            shutil.copy(source_file, destination_file)


def save_to_database(j_type):
    src_dir = 'datasets/prediction'

    match j_type:
        case 'non-predatory':
            copy_files(src_dir, 'database/non-predatory')
        case 'potentially predatory':
            copy_files(src_dir, 'database/potentially-predatory')
        case 'predatory':
            copy_files(src_dir, 'database/predatory')

    print("Saved successfully.")


if __name__ == "__main__":
    main()

"""
 Author: Viktória Lukáčová
"""
