import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',)
parser.add_argument('--output_dir',)
parser.add_argument('--min_df', type=int, default=5)
args = parser.parse_args()

texts = list()
with open(os.path.join(args.data_path, 'texts.txt')) as file:
    for line in file:
        texts.append(line.strip())

vectorizer = CountVectorizer(min_df=args.min_df)
bow_matrix = vectorizer.fit_transform(texts).toarray()

idx = np.where(bow_matrix.sum(axis=-1) > 0)
bow_matrix = bow_matrix[idx]

vocab = vectorizer.get_feature_names()

print("===>saving files")

os.makedirs(args.output_dir, exist_ok=True)

scipy.sparse.save_npz(os.path.join(args.output_dir, 'bow_matrix.npz'), scipy.sparse.csr_matrix(bow_matrix))
with open(os.path.join(args.output_dir, 'vocab.txt'), 'w') as file:
    for line in vocab:
        file.write(line + '\n')

print('===>done.')
