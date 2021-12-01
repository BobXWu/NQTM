# Code for Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder

**[EMNLP2020](https://www.aclweb.org/anthology/2020.emnlp-main.138.pdf)** | **[Presentation Video](https://slideslive.com/38938639/short-texts-topic-modeling-with-topic-distribution-quantization-and-negative-sampling-decoder)**


## Usage
### 0. Prepare environment

requirements:

    python==3.6
    tensorflow-gpu==1.13.1
    scipy==1.5.2
    scikit-learn==0.23.2 


### 1. Prepare data

Note: the data in the path ./data has been preprocessed with tokenization, filtering non-Latin characters, etc before.

    python utils/preprocess.py --data_path data/stackoverflow --output_dir input/stackoverflow


### 2. Run the model

    python run_NQTM.py --data_dir input/stackoverflow --output_dir output


### 3. Evaluation

topic coherence: [coherence score](https://github.com/dice-group/Palmetto).

topic diversity:

    python utils/TU.py --data_path output/top_words_T15_K50_1th


## Citation

If you want to use our code, please cite as

    @inproceedings{wu2020short,
        title = "Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder",
        author = "Wu, Xiaobao  and
        Li, Chunping  and
        Zhu, Yan  and
        Miao, Yishu",
        booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.emnlp-main.138",
        doi = "10.18653/v1/2020.emnlp-main.138",
        pages = "1772--1782"
    }
