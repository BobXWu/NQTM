import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()
    return args

def TU_eva(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()

    TU = 0.0
    TF = counter.sum(axis=0)
    cnt = TF * (counter > 0)

    for i in range(K):
        TU += (1 / cnt[i][np.where(cnt[i] > 0)]).sum() / T
    TU /= K

    return TU

def TU_read(data_path):
    texts = list()
    with open(os.path.join(data_path)) as file:
        for line in file:
            texts.append(line.strip())
    TU = TU_eva(texts)
    print("===>TU: {:5f}".format(TU))

if __name__ == "__main__":
    args = parse_args()
    TU_read(args.data_path)
