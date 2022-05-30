#make sure you have ktrain: !pip install ktrain
#make sure you have tfkeras: !pip install https://github.com/amaiya/eli5/archive/refs/heads/tfkeras_0_10_1.zip
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
from transformers import *
import tensorflow as tf

path = '../../datasets/original/'

exec(open('../../utils/utils.py').read())
exec(open('../../utils/evaluate.py').read())
exec(open('../../utils/embeddings.py').read())

# getting the ontology
ontology = generate_ontology('../../utils/go.obo', specific_space=True, name_specific_space='molecular_function')

# reading csv
train = pd.read_csv(path + 'train.csv')
val = pd.read_csv(path + 'val.csv')
test = pd.read_csv(path + 'test.csv')

# getting training, validation and testing features
X_train, y_train, positions_train = generate_data(train)
X_val, y_val, positions_val = generate_data(val)
X_test, y_test, positions_test = generate_data(test)
ontologies_names = test.columns[2:].values

generate_embeddings('Rostlab/prot_bert_bfd', 'prot_bert_bfd')
generate_embeddings('Rostlab/prot_bert', 'prot_bert')