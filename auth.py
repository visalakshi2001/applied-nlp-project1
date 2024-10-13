import streamlit as st
import cohere
from pinecone import Pinecone, ServerlessSpec
import os

@st.cache_resource
def authenticate_co_pc():
    from cache import COHERE_API_KEY, PINECONE_API_KEY
    global pc 
    global co

    co = cohere.Client(COHERE_API_KEY)
    pc = Pinecone(PINECONE_API_KEY)

    return pc, co

@st.cache_data
def retrieve_similar_articles(keyword: str, k: int = 5):
    authenticate_co_pc()

    index_name = "scientific-papers"
    index = pc.Index(index_name)
    print(f"Opened index {index_name}")

    # Generate embedding for the keyword
    query_embedding = co.embed(
        texts=[keyword],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings

    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)

    similar_articles = []
    for match in results['matches']:
        similar_articles.append({
            "paper_id": match['id'],
            "pmcid": match["metadata"]["paper_pmic"],
            "pmid": match["metadata"]["paper_pmid"],
            "title": match['metadata']['title'],
            "score": match['score']
        })

    return similar_articles
