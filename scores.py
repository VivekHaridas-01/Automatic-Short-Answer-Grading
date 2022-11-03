import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from bert_embedding import BertEmbedding
from allennlp.commands.elmo import ElmoEmbedder
from transformers import RobertaTokenizer, RobertaModel
from transformers import  XLNetTokenizer, XLNetModel
import torch
import scipy
import string
import tensorflow as tf
import tensorflow_hub as hub

# Preprocessing function(Stopword/Punctuation removal and question demoting)
def preprocessing(q, ans):
    stop_words = set(stopwords.words('english'))
    q_res = q.translate(str.maketrans('','',string.punctuation))
    ans_res = ans.translate(str.maketrans('','',string.punctuation))
    q_tokens = word_tokenize(q_res)
    ans_tokens = word_tokenize(ans_res)
    demoted_tokens = [t for t in ans_tokens if t not in q_tokens]
    filtered_sent = [w for w in demoted_tokens if not w in stop_words]
    return filtered_sent


def check_tokens(sent):
    if not list:
        sent = word_tokenize(sent)
    return sent

def bert(sent):
    tokens = check_tokens(sent)
    embedding = BertEmbedding().embedding(sentences=tokens)
    word_arr = []
    for i in range(len(embedding)):
        word_arr.append(embedding[i][1][0])
    return word_arr

def gpt(sent):
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'openai-gpt')
    model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'openai-gpt')
    
    tokens = check_tokens(sent)
    tokens_i = tokenizer.convert_tokens_to_ids(tokens)
    tokens_t = torch.tensor([tokens_i])
    embedding = model(tokens_t)
    
    word_arr = []
    for i in range(embedding[0].shape[1]):
        word_arr.append(embedding[0][0][i].tolist())
    return word_arr

def elmo(sent):
    tokens = check_tokens(sent)
    embedding = ElmoEmbedder().embed_sentence(tokens)
    word_arr = []

    for i in range(len(embedding[2])):
        word_arr.append(embedding[0][i])
    return word_arr

def gpt2(sent):
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
    model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'gpt2')

    tokens=check_tokens(sent)
    tokens_i = tokenizer.convert_tokens_to_ids(tokens)
    tokens_t = torch.tensor([tokens_i])
    embedding = model(tokens_t)

    word_arr = []
    for i in range(embedding[0].size()[1]):
        word_arr.append(embedding[0][0][i].tolist())
    return word_arr

def roberta(sent):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")

    tokens = check_tokens(sent)
    tokens_i = tokenizer.convert_tokens_to_ids(tokens)
    tokens_t = torch.tensor([tokens_i])
    embedding = model(tokens_t)

    word_arr = []
    for i in range(embedding[0].shape[1]):
        word_arr.append(embedding[0][0][i].tolist())
    return word_arr


def xlnet(sent):
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetModel.from_pretrained("xlnet-base-cased")

    tokens = check_tokens(sent)
    tokens_i = tokenizer.convert_tokens_to_ids(tokens)
    tokens_t = torch.tensor([tokens_i])
    embedding = model(tokens_t)

    word_arr = []
    for i in range(embedding[0].size()[1]):
        word_arr.append(embedding[0][0][i].tolist())
    return word_arr

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
univ_model = hub.load(module_url)
def universal(sent):
    tokens = check_tokens(sent)
    embedding = tf.nn.l2_normalize(univ_model(tokens))
    word_arr=[]
    for i in range(len(embedding)):
        word_arr.append(embedding[i].numpy())
    return word_arr


# Feature Extraction
df = pd.read_csv('dataset/mohler_dataset_edited.csv')
# Get  student answers from dataset
student_answers = df['student_answer'].to_list()
similarity_scores = {}
model = str(
    input('Enter a model name(bert, gpt, elmo, gpt2, roberta, xlnet, universal) to get similarity scores for: '))

# Calculate cosine similarity score for each answer
for ans in student_answers:
    q = df.loc[df['student_answer'] == ans, 'question'].iloc[0]
    model_ans = df.loc[df['student_answer'] == ans, 'desired_answer'].iloc[0]

    # Preprocess student answer
    model_preproc = preprocessing(q, model_ans)
    stu_preproc = preprocessing(q, ans)
    if len(model_preproc) == 0:
        model_preproc = model_ans.split()
    if len(stu_preproc) == 0:
        stu_preproc = ans.split()

    # Calculate and save similarity score
    if model=="bert":
        model_arr = bert(model_preproc)
        stu_arr = bert(stu_preproc)
        similarity_scores[ans]=1-scipy.spatial.distance.cosine(sum(model_arr),sum(stu_arr))

    elif model =="gpt2":
        model_arr = gpt2(model_preproc)
        stu_arr = gpt2(stu_preproc)

        e1 = [sum(i) for i in zip(*model_arr)]
        e2 = [sum(i) for i in zip(*stu_arr)]
        similarity_scores[ans] = 1-scipy.spatial.distance.cosine(e1,e2)

    elif model=="gpt":
        model_arr = gpt(model_preproc)
        stu_arr = gpt(stu_preproc)
        e1 = [sum(i) for i in zip(*model_arr)]
        e2 = [sum(i) for i in zip(*stu_arr)]
        similarity_scores[ans] = 1-scipy.spatial.distance.cosine(e1,e2)

    elif model=="elmo":
        model_arr = elmo(model_preproc)
        stu_arr = elmo(stu_preproc)
        similarity_scores[ans] = 1-scipy.spatial.distance.cosine(sum(model_arr), sum(stu_arr))
    
 
    elif model == "xlnet":
        model_arr = xlnet(model_preproc)
        stu_arr = xlnet(stu_preproc)
        e1 = [sum(i) for i in zip(*model_arr)]
        e2 = [sum(i) for i in zip(*stu_arr)]
        similarity_scores[ans] = 1 - scipy.spatial.distance.cosine(e1, e2)

    elif model == "roberta":
        model_arr = roberta(model_preproc)
        stu_arr = roberta(stu_preproc)
        e1 = [sum(i) for i in zip(*model_arr)]
        e2 = [sum(i) for i in zip(*stu_arr)]
        similarity_scores[ans] = 1 - scipy.spatial.distance.cosine(e1, e2)
        print(list(similarity_scores)[-1])
    
    elif model=="universal":
        model_arr=universal(model_preproc)
        stu_arr=universal(stu_preproc)
        similarity_scores[ans]=1-scipy.spatial.distance.cosine(sum(model_arr),sum(stu_arr))

col_name = model+"_similarity_score"
for a in student_answers:
    df.loc[df['student_answer'] == a, col_name] = similarity_scores[a]

# Apply normalization techniques
columns = ['bert_similarity_score','elmo_similarity_score','gpt_similarity_score','gpt2_similarity_score','roberta_similarity_score','xlnet_similarity_score'.'universal_similarity_score']
for c in columns:
    df['normalized_'+c] = MinMaxScaler().fit_transform(np.array(df[c]).reshape(-1,1))
df.to_csv('dataset/answers_with_similarity_score.csv')
