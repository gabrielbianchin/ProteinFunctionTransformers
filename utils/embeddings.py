from tqdm import tqdm

def create_embeddings(cur_emb, mode):
    cur_emb = np.array(cur_emb)
    embeddings = []
    if mode == 'min':
        embeddings = cur_emb.min(0)
    elif mode == 'max':
        embeddings = cur_emb.max(0)
    elif mode == 'med':
        embeddings = np.median(cur_emb, axis=0)
    else:
        embeddings = np.mean(cur_emb, axis=0)
    return embeddings

def merge_embeddings(X_val, y_val, positions, mode, model, tokenizer):
    merged_emb = []
    last_pos = positions[0]
    cur_emb = []
    y_val_merged = [y_val[0]]
    
    for i in tqdm(range(len(X_val))):
        cur_pos = positions[i]
        if last_pos == cur_pos:
            # gera embeddings
            input_ids = tf.constant(tokenizer.encode(X_val[i]))[None, :]
            outputs = model(input_ids)
            last_hidden_states = outputs[0]

            cur_emb.append(last_hidden_states.numpy()[0, 0, :])

        else:
            # Salvar as predições
            merged_emb.append(create_embeddings(cur_emb, mode))

            # Resetar estado para próxima predição
            last_pos = cur_pos

            # gera embeddings
            input_ids = tf.constant(tokenizer.encode(X_val[i]))[None, :]
            outputs = model(input_ids)
            last_hidden_states = outputs[0]

            cur_emb = [last_hidden_states.numpy()[0, 0, :]]
            y_val_merged.append(y_val[i])

    # Juntar último valor
    merged_emb.append(create_embeddings(cur_emb, mode))
    
    return np.array(merged_emb, dtype='float64'), np.array(y_val_merged)

def save_embeddings_and_y_merged(model_name_short, X, y, positions, model, tokenizer, set_name):
    emb_merged, y_merged = merge_embeddings(X, y, positions, 'avg', model, tokenizer)
    
    # save embeddings and y_merged
    with open('../../embeddings/' + model_name_short + '/embedding_merged_' + set_name + '.npy', 'wb') as f:
        np.save(f, emb_merged)
        
    with open('../../y_merged/y_merged_' + set_name + '.npy', 'wb') as f:
        np.save(f, y_merged)
    
def generate_embeddings(model_name, model_name_short):
    # get model
    t = text.Transformer(model_name, maxlen=100, classes=ontologies_names)
    trn = t.preprocess_train(X_train, y_train)
    val = t.preprocess_test(X_val, y_val)
    model = t.get_classifier()
    model.load_weights(model_name)
    
    # get learner
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)
    learner.model.save_pretrained('/content/model')
    
    # get tokenizer and load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained('/content/model')
    
    # save embeddings and y_merged
    save_embeddings_and_y_merged(model_name_short, X_train, y_train, positions_train, model, tokenizer, 'train')
    save_embeddings_and_y_merged(model_name_short, X_val, y_val, positions_val, model, tokenizer, 'val')
    save_embeddings_and_y_merged(model_name_short, X_test, y_test, positions_test, model, tokenizer, 'test')