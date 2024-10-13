import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification



@st.cache_resource
def load_literature():
    papers = pd.concat([pd.read_csv("/content/alzheimer_biomarker_1000.csv"), 
                        pd.read_csv("/content/alzheimer_biomarker_2000.csv")]) \
                            .dropna(subset=["fulltext"]).reset_index(drop=True) \
                                .astype(str)
    return papers

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer 