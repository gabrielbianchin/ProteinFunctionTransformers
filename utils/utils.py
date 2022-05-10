import ktrain
from ktrain import text

def get_ancestors(ontology, term):
  list_of_terms = []
  list_of_terms.append(term)
  data = []
  
  while len(list_of_terms) > 0:
    new_term = list_of_terms.pop(0)

    if new_term not in ontology:
      break
    data.append(new_term)
    for parent_term in ontology[new_term]['parents']:
      if parent_term in ontology:
        list_of_terms.append(parent_term)
  
  return data

def generate_ontology(file, specific_space=False, name_specific_space=''):
  ontology = {}
  gene = {}
  flag = False
  with open(file) as f:
    for line in f.readlines():
      line = line.replace('\n','')
      if line == '[Term]':
        if 'id' in gene:
          ontology[gene['id']] = gene
        gene = {}
        gene['parents'], gene['alt_ids'] = [], []
        flag = True
        
      elif line == '[Typedef]':
        flag = False
      
      else:
        if not flag:
          continue
        items = line.split(': ')
        if items[0] == 'id':
          gene['id'] = items[1]
        elif items[0] == 'alt_id':
          gene['alt_ids'].append(items[1])
        elif items[0] == 'namespace':
          if specific_space:
            if name_specific_space == items[1]:
              gene['namespace'] = items[1]
            else:
              gene = {}
              flag = False
          else:
            gene['namespace'] = items[1]
        elif items[0] == 'is_a':
          gene['parents'].append(items[1].split(' ! ')[0])
        elif items[0] == 'name':
          gene['name'] = items[1]
        elif items[0] == 'is_obsolete':
          gene = {}
          flag = False
    
    key_list = list(ontology.keys())
    for key in key_list:
      ontology[key]['ancestors'] = get_ancestors(ontology, key)
      for alt_ids in ontology[key]['alt_ids']:
        ontology[alt_ids] = ontology[key]
    
    for key, value in ontology.items():
      if 'children' not in value:
        value['children'] = []
      for p_id in value['parents']:
        if p_id in ontology:
          if 'children' not in ontology[p_id]:
            ontology[p_id]['children'] = []
          ontology[p_id]['children'].append(key)
    
  return ontology

def get_and_print_children(ontology, term):
  children = {}
  if term in ontology:
    for i in ontology[term]['children']:
      children[i] = ontology[i]
      print(i, ontology[i]['name'])
  return children


def generate_data(df, subseq=100):
  X = []
  y = []
  positions = []
  sequences = df.iloc[:, 1].values

  for i in tqdm(range(len(sequences))):
    seq = []
    seq.append(' '.join(list(sequences[i])))
    number_of_seqs = int(np.ceil(len(sequences[i]) / subseq))

    for idx in range(number_of_seqs):
        if idx != number_of_seqs - 1:
            X.append(' '.join(list(sequences[i][idx * subseq : (idx + 1) * subseq])))
        else:
            X.append(' '.join(list(sequences[i][idx * subseq : ])))
        positions.append(i)
        y.append(df.iloc[i, 2:])
  
  return X, np.array(y, dtype=int), positions


def train_and_save_model(model_name, MAX_LEN, ontologies_names, path=''):
  t = text.Transformer(model_name, maxlen=MAX_LEN, classes=ontologies_names)
  trn = t.preprocess_train(X_train, y_train)
  val = t.preprocess_test(X_val, y_val)

  model = t.get_classifier()
  learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

  learner.autofit(1e-5, 10, early_stopping=1, checkpoint_folder=path)


def create_prediction(cur_pred, mode):
  cur_pred = np.array(cur_pred)
  prediction = []
  if mode == 'min':
    prediction = cur_pred.min(0)
  elif mode == 'max':
    prediction = cur_pred.max(0)
  elif mode == 'med':
    prediction = np.median(cur_pred, axis=0)
  else:
    prediction = np.mean(cur_pred, axis=0)
  return prediction.tolist()


def merge_predictions(predictions, y_val, positions, mode):
  merged_predictions = []
  last_pos = positions[0]
  cur_pred = []
  y_val_merged = [y_val[0]]
    
  for i in range(len(predictions)):
    cur_pos = positions[i]
    if last_pos == cur_pos:
      cur_pred.append(predictions[i])
    else:
      merged_predictions.append(create_prediction(cur_pred, mode))
      last_pos = cur_pos
      cur_pred = [predictions[i]]
      y_val_merged.append(y_val[i])

  merged_predictions.append(create_prediction(cur_pred, mode))
    
  return merged_predictions, y_val_merged