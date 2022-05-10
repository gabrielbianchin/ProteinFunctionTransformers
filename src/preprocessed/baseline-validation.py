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
X_train, y_train = train['sequences'].values, train.iloc[:, 2:].values
X_val, y_val = val['sequences'].values, val.iloc[:, 2:].values
X_test, y_test = test['sequences'].values, test.iloc[:, 2:].values
ontologies_names = test.columns[2:].values

# predicting
proba = np.sum(y_train, axis=0) / len(y_train)
proba = np.expand_dims(proba, axis=0)
predictions = np.repeat(proba, len(val), axis=0)
gt = y_val
evaluate(predictions, gt)