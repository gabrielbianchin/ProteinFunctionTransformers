#make sure you have ktrain: !pip install ktrain
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

# getting training, validation and testing features
X_train, y_train, positions_train = generate_data(train)
X_val, y_val, positions_val = generate_data(val)
X_test, y_test, positions_test = generate_data(test)
ontologies_names = test.columns[2:].values

# predict with ProtBertBfd
t = text.Transformer('Rostlab/prot_bert_bfd', maxlen=100, classes=ontologies_names)
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_val, y_val)
model = t.get_classifier()
model.load_weights('../../weights/preprocessed/prot_bert_bfd/weights-xx.hdf5') # change the weights
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
predictor = ktrain.get_predictor(learner.model, preproc=t)
predict_prot_bert_bfd = predictor.predict_proba(X_val)

# predict with ProtBert
t = text.Transformer('Rostlab/prot_bert', maxlen=100, classes=ontologies_names)
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_val, y_val)
model = t.get_classifier()
model.load_weights('../../weights/preprocessed/prot_bert/weights-xx.hdf5') # change the weights
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
predictor = ktrain.get_predictor(learner.model, preproc=t)
predict_prot_bert = predictor.predict_proba(X_val)

# merge predictions of broken proteins
predict_prot_bert_bfd_avg, y_val_merged = merge_predictions(predict_prot_bert_bfd, y_val, positions_val, 'avg')
predict_prot_bert_avg, y_val_merged = merge_predictions(predict_prot_bert, y_val, positions_val, 'avg')

# ensemble
predictions_avg = np.mean(np.array([ predict_prot_bert_bfd_avg, predict_prot_bert_avg ]), axis=0)
predictions_max = np.maximum(predict_prot_bert_bfd_avg, predict_prot_bert_avg)
predictions_min = np.minimum(predict_prot_bert_bfd_avg, predict_prot_bert_avg)

# evaluate
evaluate(predictions_avg, y_val_merged)
evaluate(predictions_max, y_val_merged)
evaluate(predictions_min, y_val_merged)