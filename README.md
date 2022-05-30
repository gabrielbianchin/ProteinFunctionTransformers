# Prediction of Protein Molecular Functions Using Transformers

## Introduction
At the end of 2021, there were more than 200 million proteins in which their molecular functions were still unknown. As the empirical determination of these functions is slow and expensive, several research groups around the world have applied machine learning to perform the prediction of protein functions. In this work, we evaluate the use of Transformer architectures to classify protein molecular functions. Our classifier uses the embeddings resulting from two Transformer-based architectures as input to a Multi-Layer Perceptron classifier. This model got Fmax of 0.562 in our database and, when we applied this model to the same database used by DeepGOPlus, we reached the value of 0.617, surpassing the best result available in the literature.

## Reproducibility
For Original and Preprocessed datasets:
```
1. Run prot_bert.py file
2. Run prot_bert_bfd.py file
3. Run embeddings.py file
4. Run deep-features.py file
```

## Citation
This repository contains the source codes of Prediction of Protein Molecular Functions Using Transformers, as given in the paper:

Felipe Lopes de Mello, Gabriel Bianchin de Oliveira, Helio Pedrini, Zanoni Dias. "Prediction of Protein Molecular Functions Using Transformers", in proceedings of the 21st International Conference on Artificial Intelligence and Soft Computing (ICAISC). Virtual Conference, June 19 - 23, 2022.

If you use this source code and/or its results, please cite our publication:
```
@inproceedings{MELLO_2022_ICAISC,
  author = {F.L. Mello and G.B. Oliveira and H. Pedrini and Z. Dias},
  title = {{Prediction of Protein Molecular Functions Using Transformers}},
  booktitle = {21st International Conference on Artificial Intelligence and Soft Computing (ICAISC)},
  address = {Virtual Conference},
  month = jun,
  year = {2022}
}
```