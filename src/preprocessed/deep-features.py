#make sure you have ktrain: !pip install ktrain
#make sure you have tfkeras: !pip install https://github.com/amaiya/eli5/archive/refs/heads/tfkeras_0_10_1.zip
import pandas as pd
import numpy as np
import ktrain
from ktrain import text

path = '../../datasets/preprocessed/'

exec(open('../../utils/utils.py').read())
exec(open('../../utils/evaluate.py').read())

# getting the ontology
ontology = generate_ontology('../../utils/go.obo', specific_space=True, name_specific_space='molecular_function')

# reading csv
train = pd.read_csv(path + 'train.csv')
val = pd.read_csv(path + 'val.csv')
test = pd.read_csv(path + 'test.csv')

# getting training and validation features
ontologies_names = test.columns[2:].values
y_val_merged = np.load('../../y_merged/y_merged_val.npy')
y_test_merged = np.load('../../y_merged/y_merged_test.npy')

# load embeddings
emb_prot_bert_merged_train = np.load('../../embeddings/prot_bert/embedding_merged_train.npy')
emb_prot_bert_merged_val = np.load('../../embeddings/prot_bert/embedding_merged_val.npy')
emb_prot_bert_bfd_merged_train = np.load('../../embeddings/prot_bert_bfd/embedding_merged_train.npy')
emb_prot_bert_bfd_merged_val = np.load('../../embeddings/prot_bert_bfd/embedding_merged_val.npy')

# merge embeddings
embeddings_train = np.concatenate((emb_prot_bert_merged_train, emb_prot_bert_bfd_merged_train), axis=1)
embeddings_val = np.concatenate((emb_prot_bert_merged_val, emb_prot_bert_bfd_merged_val), axis=1)

# predict
predict = cria_e_treina_mlp(embeddings_train, y_train_merged, (embeddings_val, y_val_merged), 1000, 3, '../../deep_features/concat_merged_1000_3')

# evaluate
evaluate(predict, y_val_merged)