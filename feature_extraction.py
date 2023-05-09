import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
import pandas as pd
from bert_embedding import BertEmbedding
from transformers import RobertaTokenizer, RobertaModel, XLNetTokenizer, XLNetModel
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel,GPT2LMHeadModel,GPT2Tokenizer
import torch
import scipy
import string
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Preprocessing function(Stopword/Punctuation removal and question demoting)
def preprocessing(q, ans):
    q = q.lower()
    ans = ans.lower()
    stop_words = set(stopwords.words('english'))
    q_res = q.translate(str.maketrans('', '', string.punctuation))
    ans_res = ans.translate(str.maketrans('', '', string.punctuation))
    q_tokens = word_tokenize(q_res)
    ans_tokens = word_tokenize(ans_res)
    demoted_tokens = [t for t in ans_tokens if t not in q_tokens]
    filtered_sent = [w for w in demoted_tokens if not w in stop_words]
    return filtered_sent

def check_tokens(sent):
    if not list:
        sent = word_tokenize(sent)
    return sent

# Embedding Models to extract Feature 1 - semantic similarity between answer embeddings
def bert(sent):
    tokens = check_tokens(sent)
    embedding = BertEmbedding().embedding(sentences=tokens)
    word_arr = []
    for i in range(len(embedding)):
        word_arr.append(embedding[i][1][0])
    return word_arr


def gpt1(sent):
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids)
        embed = outputs[0][:, -1, :]
    embed=embed.numpy()
    return np.squeeze(embed)

def gpt2(sent):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
    input_ids = tokenizer.encode_plus(sent, return_tensors='pt', add_special_tokens=True)

    with torch.no_grad():
        outputs = model(**input_ids)
        last_hidden_states = outputs.hidden_states[-1]
        embed = torch.mean(last_hidden_states, dim=1).squeeze()
    return embed

def elmo(tokens):
    elmo = hub.load("https://tfhub.dev/google/elmo/3").signatures["default"]
    sent = [" ".join(tokens)]
    word_arr = elmo(tf.constant(sent))["elmo"]
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

def universal(sent):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    univ_model = hub.load(module_url)
    tokens = check_tokens(sent)
    embedding = tf.nn.l2_normalize(univ_model(tokens))
    word_arr=[]
    for i in range(len(embedding)):
        word_arr.append(embedding[i].numpy())
    return word_arr


# Feature 2 - Keyword Match Ratio
def keyword_match(s_ans, m_ans):
    m = Rake()
    m.extract_keywords_from_text(m_ans)
    ma = m.get_ranked_phrases()
    s = Rake()
    s.extract_keywords_from_text(s_ans)
    sa = s.get_ranked_phrases()
    cnt = 0
    for i in ma:
        if i in sa:
            cnt += 1
    match = cnt/len(ma)
    return match

#Feature 3 - Length Ratio
def preprocessing_lr(ans):
    stop_words = set(stopwords.words('english'))
    ans_res = ans.translate(str.maketrans('','',string.punctuation))
    ans_tokens = word_tokenize(ans_res)
    filtered_sent = [w for w in ans_tokens if not w in stop_words]
    return filtered_sent



# Feature Extraction
df = pd.read_csv('Dataset 1.csv')
# Similarity score feature extraction
# Get  student answers from dataset
student_answers = df['student_answer'].to_list()
similarity_scores = {}
model = str(
    input('Enter a model name(bert, gpt, elmo, gpt2, roberta, xlnet, universal) to get similarity scores for: '))

# Calculate cosine similarity score for each answer
for ans in student_answers:
    q = df.loc[df['student_answer'] == ans, 'question'].iloc[0]
    model_ans = df.loc[df['student_answer'] == ans, 'desired_answer'].iloc[0]
    if ans == 'Not Answered':
        similarity_scores[ans] = 0
    else:
        # Preprocess student answer
        model_preproc = preprocessing(q, model_ans)
        stu_preproc = preprocessing(q, ans)
        if len(model_preproc) == 0:
            model_preproc = model_ans.split()
        if len(stu_preproc) == 0:
            stu_preproc = ans.split()

        # Calculate and save similarity score based on the model
        if model=="bert":
            model_arr = bert(model_preproc)
            stu_arr = bert(stu_preproc)
            similarity_scores[ans]=1-scipy.spatial.distance.cosine(sum(model_arr),sum(stu_arr))
            
        elif model == "gpt":
            if model_preproc=='-':
                model_preproc = "N/A"
            if stu_preproc == list('-'):
                stu_preproc = "N/A"
            model_arr = gpt1(" ".join(model_preproc))
            stu_arr = gpt1(" ".join(stu_preproc))
            similarity_scores[ans]=1-scipy.spatial.distance.cosine(model_arr,stu_arr)

        elif model=="gpt2":
            if model_preproc=='-':
                model_preproc = "N/A"
            if stu_preproc == list('-'):
                stu_preproc = "N/A"
            model_arr = gpt2(" ".join(model_preproc))
            stu_arr = gpt2(" ".join(stu_preproc))
            similarity_scores[ans]=1-scipy.spatial.distance.cosine(model_arr,stu_arr)

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
        
        elif model=="universal":
            model_arr=universal(model_preproc)
            stu_arr=universal(stu_preproc)
            similarity_scores[ans]=1-scipy.spatial.distance.cosine(sum(model_arr),sum(stu_arr))

col_name = model+"_similarity_score"
for a in student_answers:
    df.loc[df['student_answer'] == a, col_name] = similarity_scores[a]

#Normalize similarity scores and save to file
df['normalized_'+col_name] = MinMaxScaler().fit_transform(np.array(df[col_name]).reshape(-1,1))

feature = str(
    input("Enter a feature(keywordmatch, lengthratio) to extract, or any character to just continue: "))

if feature == 'keywordmatch':
    # Keyword Match feature extraction
    match_val = {}
    for ind in df.index:
        ans=df['student_answer'][ind]
        model_ans = df['desired_answer'][ind]
        match_val[ind]=keyword_match(ans,model_ans)

    c = "keyword_match"
    for ind in df.index:
        df.at[ind, c] = match_val[ind]

elif feature=='lengthratio':
    #Length Ratio feature extraction
    length_ratio = {}
    for ind in df.index:
        ans=df['student_answer'][ind]
        model_ans = df['desired_answer'][ind]
        stu_preproc=preprocessing_lr(ans)
        model_preproc=preprocessing_lr(model_ans)
        if len(model_preproc) == 0:
            model_preproc = model_ans.split()
        if len(stu_preproc) == 0:
            stu_preproc = ans.split()
        length_ratio[ind]=len(stu_preproc)/len(model_preproc)

    c = "length_ratio"
    for ind in df.index:
        df.at[ind, c] = length_ratio[ind]
    # Apply normalization techniques
    df['normalized_length_ratio'] = MinMaxScaler().fit_transform(np.array(df[c]).reshape(-1,1))


df.to_csv('Dataset 1.csv')
