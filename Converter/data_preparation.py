import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('csv/HITECH.csv',
                 usecols=['Title', 'Author', 'Affiliation', 'Abstract', 'Source', 'Keywords', 'References', 'Text'])

# Replace NaN values with an empty string
df = df.fillna('')

# Initialize a dictionary to store the sequences and words for each column
data_dict = {}

# Tokenize and generate sequences for each column
for column in df.columns:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df[column])
    sequences = tokenizer.texts_to_sequences(df[column])
    padded_sequences = pad_sequences(sequences)

    # Get the word index for the tokenizer
    word_index = tokenizer.word_index

    # Invert the word index to get a dictionary that maps indices to words
    index_to_word = {index: word for word, index in word_index.items()}

    # Convert the sequences to their corresponding words
    word_sequences = []
    for seq in sequences:
        words = [index_to_word[idx] for idx in seq]
        word_sequences.append(words)

    # Store the padded sequences and word sequences in the dictionary
    data_dict[column] = {'padded_sequences': padded_sequences, 'word_sequences': word_sequences}

# Save the data to a file
np.savez_compressed('data.npz', **data_dict)

print(data_dict['Title'])
