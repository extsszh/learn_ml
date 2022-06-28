import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util

def get_median_similarity(keyword, topics):
    model_name = 'sentence-transformers/bert-base-nli-mean-tokens'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = {
        'input_ids' : [],
        'attention_mask' : []
    }

    topics = [keyword] + topics

    for topic in topics:
        new_tokens = tokenizer.encode_plus(topic, 
                            max_length = 128, 
                            truncation = True, 
                            padding = 'max_length', 
                            return_tensors = 'pt')
        
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)

    embeddings = outputs.last_hidden_state

    mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.shape).float()

    mask_embeddings = embeddings * mask

    summed = torch.sum(mask_embeddings, 1)
    counts = torch.clamp(mask.sum(1), min = 1e-9)
    mean_pooled = summed / counts

    mean_pooled = mean_pooled.detach().numpy()

    return(np.median(cosine_similarity([mean_pooled[0]], mean_pooled[1:])[0]))

def get_median_similarity_2(keyword, topics):

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    keyword_embedding = model.encode(keyword)

    similarities = []
    for topic in topics:
        similarities.append(util.pytorch_cos_sim(keyword_embedding, model.encode(topic)).numpy()[0])

    return (np.median(similarities))

def get_relevant_articles():
    with open('articles_2020_2022.pickle', 'rb') as f:
        data = pickle.load(f)[:50]

    relevant_articles = []

    for article in data:
        try:
            if get_median_similarity_2('software development', article['index_terms']['ieee_terms']['terms']) > 0.2:
                relevant_articles.append(article)
        except:
            continue
    
    with open('relevant_articles_2020_2022_2.pickle', 'wb') as f:
        pickle.dump(relevant_articles, f)

def visualize_articles():

    with open(f'relevant_articles_2020_2022_2.pickle', 'rb') as f:
        data = pickle.load(f)
    
    term_frequency = dict()
    for article in data:
        try:
            for term in article['index_terms']['ieee_terms']['terms']:
                term_frequency[term] = term_frequency[term] + article['citing_paper_count'] + 1 if term in term_frequency else article['citing_paper_count'] + 1
        except:
            continue
    
    df = pd.DataFrame({
        'Term': term_frequency.keys(),
        'Frequency' : term_frequency.values()
    })
    df = df.sort_values(by=['Frequency'], ascending = False)
    plt.barh(df['Term'][0:25], df['Frequency'][0:25])
    plt.yticks(rotation=30, fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("Paper Citation Count", fontsize=36)
    plt.show()

# topic frequency visualization
if __name__ == '__main__':
    get_relevant_articles()
    visualize_articles()