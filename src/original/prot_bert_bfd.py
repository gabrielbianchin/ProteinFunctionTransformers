import pandas as pd
import numpy as np
import ktrain
from ktrain import text

path = '../../datasets/original/'

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

# training model
train_and_save_model(model_name='Rostlab/prot_bert_bfd', MAX_LEN=100, ontologies_names=ontologies_names, path='../../weights/original/prot_bert_bfd')

# predicting
model_name = 'Rostlab/prot_bert_bfd'
t = text.Transformer(model_name, maxlen=100, classes=ontologies_names)
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_val, y_val)
model = t.get_classifier()
model.load_weights('../../weights/original/prot_bert_bfd/weights-xx.hdf5') # change the weights
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
predictor = ktrain.get_predictor(learner.model, preproc=t)
predict = predictor.predict_proba(X_test)

# merging predictions
predictions_avg, y_merged = merge_predictions(predict, y_test, positions_test, 'avg')
evaluate(predictions_avg, y_merged)