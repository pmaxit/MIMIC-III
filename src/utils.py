from fastai2.text.all import *
from fastai2.text.data import *
import joblib
import dill
from transformers import BertTokenizer, BertForPreTraining
from transformers import BertConfig, BertForSequenceClassification, BertModel

bert_tok = BertTokenizer.from_pretrained(
 "bert-base-uncased",
)

fastai_bert_vocab =L(bert_tok.vocab.keys())



class FastAIBertTokenizer(Transform):
    """ Wrapper around Bert Tokenizer to be compatible with Fastai 
    
        Bert Tokenization adds ## extra symbol to handle inner word embeddings
        embeddings:  em ## bed ##dings
    """
    def __init__(self, tokenizer: BertTokenizer, split_char = " ", max_seq_len:int = 5000, **kwargs):
        self.max_seq_len = max_seq_len
        self.split_char = split_char
        self.tokenizer = tokenizer
        self.sentence_idx = 0


    def encodes(self, sentence):
        result = self.tokenizer.encode_plus(sentence,return_tensors='pt', 
                    max_length = self.max_seq_len, pad_to_max_length=True,)
        
        # Needed to add index because by default it returns list of list
        # in our case, we only have one sentence
        return (TensorText(result['input_ids'][0]) , result['attention_mask'][0])

    def decodes(self, tokens):
        return TitledStr(self.tokenizer.decode(tokens[0],skip_special_tokens=True))

class MyNumericalize(Transform):
    def __init__(self):
        pass
    
    def encodes(self, x):
        return TensorText(tensor(x))
    def decodes(self, o):
        return L([o_.item() for o_ in o])


# #export
# @typedispatch
# def wandb_process(x, y, samples, outs):
#     sentences = []
#     labels = []
#     originals = []
#     for s,o in zip(samples,outs):
#         # get the sentence and label together
#         for k,v,l in zip(s[0],o[0],s[2]):
#             sentences.append(k)
#             labels.append(v)
#             originals.append(l)
#     data = [[s[0], s[1], o[0]] for s,o in zip(samples,outs)]
#     return {"Prediction Samples": wandb.Table(data=data[:100], columns=["Text", "Target", "Prediction"])}


# # it will call the show_batch with x as TensorText , y as tuples. Samples will contain the original samples
# @typedispatch
# def show_results(x, y, samples, outs, ctxs=None, max_n=9, **kwargs):
#     sentences = []
#     labels = []
#     originals = []
#     for s,o in zip(samples,outs):
#         # get the sentence and label together
#         for k,v,l in zip(s[0],o[0],s[2]):
#             sentences.append(k)
#             labels.append(v)
#             originals.append(l)
#     df = pd.DataFrame.from_dict({'Sentence': sentences,'Original' :originals,'Label':labels})
#     display_df(df)
    
#     return ctxs

# @typedispatch
# def show_batch(x:tuple, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
#     sentences = []
#     labels = []
#     for s in samples:
#         # get the sentence and label together
#         for k,v in zip(s[0],s[2]):
#             sentences.append(k)
#             labels.append(v)
#     df = pd.DataFrame.from_dict({'Sentence': sentences,'Label':labels})
#     display_df(df)
# #     if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
# #     if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
# #     ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
#     return ctxs


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