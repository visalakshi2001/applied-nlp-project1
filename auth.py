import streamlit as st
import cohere
from pinecone import Pinecone, ServerlessSpec
import os

@st.cache_resource(show_spinner=False)
def initiate_cohere():
    co = cohere.Client(api_key=st.secrets["COHERE_API_KEY"]) 
    return co  

@st.cache_resource(show_spinner=False)
def authenticate_co_pc():
    co = initiate_cohere()
    pc = Pinecone(st.secrets["PINECONE_API_KEY"])

    return pc, co

@st.cache_data(show_spinner=False)
def retrieve_similar_articles(keyword: str, k: int = 5):

    global pc
    global co
    
    pc, co = authenticate_co_pc()

    index_name = "scientific-papers"
    index = pc.Index(index_name)
    print(f"Opened index {index_name}")

    # DOCUMENTATION CODE: https://docs.cohere.com/v2/reference/embed
    query_embedding = co.embed(
        texts=[keyword],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings
    # END OF DOCUMENTATION CODE https://docs.cohere.com/v2/reference/embed
    
    # DOCUMENTATION CODE https://docs.pinecone.io/guides/get-started/quickstart
    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    # END OF DOCUMENTATION CODE https://docs.pinecone.io/guides/get-started/quickstart

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


EXAMPLE_KW = """
    "Alzheimer's disease biomarkers",
    "Early detection of Alzheimer's",
    "Blood-based biomarkers for Alzheimer's",
    "Amyloid PET imaging in Alzheimer's",
    "Tau protein as Alzheimer's biomarker",
    "Cerebrospinal fluid biomarkers in Alzheimer's",
    "Genetic risk factors for Alzheimer's",
    "Neurofilament light chain in Alzheimer's",
    "Lipid biomarkers for Alzheimer's",
    "Alzheimer's disease progression markers",
    "Non-invasive Alzheimer's diagnostic tools",
    "Molecular imaging in Alzheimer's diagnosis",
    "Cognitive decline markers in Alzheimer's",
    "Predictive biomarkers for Alzheimer's",
    "Alzheimer's disease neuroimaging initiative",
    "Proteomics in Alzheimer's research",
    "Fluid biomarkers for Alzheimer's",
    "Eye and vision changes in Alzheimer's",
    "Sleep disturbances as Alzheimer's biomarkers",
    "Standardization of Alzheimer's biomarkers"
"""