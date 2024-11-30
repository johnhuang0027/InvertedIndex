from pymongo import MongoClient
import re
from collections import defaultdict
from math import sqrt, log

client = MongoClient("mongodb://localhost:27017/")
db = client['search_engine']
documents_collection = db['documents']
terms_collection = db['terms']

# Documents
documents = {
    1: "After the medication, headache and nausea were reported by the patient.",
    2: "The patient reported nausea and dizziness caused by the medication.",
    3: "Headache and dizziness are common effects of this medication.",
    4: "The medication caused a headache and nausea, but no dizziness was reported.",
}

def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower()) #remvoes punctatuion and lowercases
    return text.split() #tokens

#iteritively join combinations of terms and return ngram
def generate_Ngrams(tokens, n=1):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_Tfidf(term_freq, doc_count, total_docs):
    tf = term_freq
    idf = log(total_docs / doc_count) if doc_count > 0 else 0
    return tf * idf

total_docs = len(documents)
inverted_index = defaultdict(lambda: defaultdict(list))

#creates ngrams to add to db
for doc_id, content in documents.items():
    tokens = tokenize(content)
    unigrams = generate_Ngrams(tokens, 1)
    bigrams = generate_Ngrams(tokens, 2)
    trigrams = generate_Ngrams(tokens, 3)
    terms = unigrams + bigrams + trigrams

    #append the docs and pos data to list
    for pos, term in enumerate(terms):
        inverted_index[term]['docs'].append(doc_id)
        inverted_index[term]['pos'].append(pos)
    
    #insert to DB
    documents_collection.insert_one({
        "_id": doc_id,
        "content": content
    })

#compute tfidf and insert into DB
for term, data in inverted_index.items():
    doc_freq = len(set(data['docs']))
    for doc_id in set(data['docs']):
        tf_idf = compute_Tfidf(data['docs'].count(doc_id), doc_freq, total_docs)
        terms_collection.insert_one({
            "term": term,
            "docs": list(set(data['docs'])),
            "pos": data['pos'],
            "tf_idf": tf_idf
        })

queries = {
    "q1": "nausea and dizziness",
    "q2": "effects",
    "q3": "nausea was reported",
    "q4": "dizziness",
    "q5": "the medication",
}

def cosine_similarity(query_vector, doc_vector):
    dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
    query_norm = sqrt(sum(q**2 for q in query_vector))
    doc_norm = sqrt(sum(d**2 for d in doc_vector))
    return dot_product / (query_norm * doc_norm) if query_norm and doc_norm else 0

# Search and Rank
results = []
for query_id, query in queries.items():
    query_tokens = tokenize(query)
    matching_docs = defaultdict(list)

    #if the docs match, add tfidf of term
    for token in query_tokens:
        term = terms_collection.find_one({"term": token})
        if term:
            for doc_id in term['docs']:
                matching_docs[doc_id].append(term['tf_idf'])

    query_vector = [1] * len(query_tokens)
    for doc_id, doc_vector in matching_docs.items():
        score = cosine_similarity(query_vector, doc_vector)
        results.append({
            "query_id": query_id,
            "document" : doc_id,
            "content": documents[doc_id],
            "score": score
        })

# sort in descending order and print
results.sort(key=lambda x: (x["query_id"], -x["score"]))
for result in results:
    print(f"Query ID: {result['query_id']}, Document ID: {result['document']}, Score: {result['score']:.2f}")
