from fastai2.text.all import *
from fastai2.text.data import *
import joblib
import dill
from transformers import BertTokenizer, BertForPreTraining
from transformers import BertConfig, BertForSequenceClassification, BertModel
import bcolz
import pickle

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix)) 
    
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim



def load_wordvectors(glove_path=""):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    
    vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))


def get_glove_matrix(glove_path="", vocab=None, emb_dim = 50):
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
        
    return weights_matrix

class FastAIBertTokenizer(Transform):
    """ Wrapper around Bert Tokenizer to be compatible with Fastai 
    
        Bert Tokenization adds ## extra symbol to handle inner word embeddings
        embeddings:  em ## bed ##dings
    """
    def __init__(self, tokenizer: BertTokenizer, split_char = " ", max_seq_len:int = 5000,  fill_to_max=True, **kwargs):
        self.max_seq_len = max_seq_len
        self.split_char = split_char
        self.tokenizer = tokenizer
        self.sentence_idx = 0
        self.fill_to_max = fill_to_max


    def encodes(self, sentence):
        result = self.tokenizer.encode_plus(sentence,return_tensors='pt', 
                    max_length = self.max_seq_len, pad_to_max_length=self.fill_to_max)
        
        # Needed to add index because by default it returns list of list
        # in our case, we only have one sentence
        if self.fill_to_max: 
            return (TensorText(result['input_ids'][0]) , result['attention_mask'][0])
        else:
            return TensorText(result['input_ids'][0])

    def decodes(self, tokens):
        return TitledStr(self.tokenizer.decode(tokens,skip_special_tokens=True))


def load_from_disk(path="test.pkl", use_dill=False):
    res = None
    if use_dill:
        with open(path, 'rb') as f:
            res = dill.load(f)
    else:
        res = joblib.load(path)
        
    return res

def dump_to_disk(obj, path="test.pkl", use_dill=False):
    if dill:
        with open(path, 'wb') as f:
            dill.dump(obj, f)
    else:
        joblib.dump(obj, path)
        
        
