import pandas as pd
import numpy as np

path = '../../datasets/preprocessed/'

exec(open('../../utils/utils.py').read())
exec(open('../../utils/evaluate.py').read())

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

# predict

# ensemble