import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import torch


@st.cache_data(show_spinner=False)
def load_literature():
    papers = pd.concat([pd.read_csv("data/alzheimer_biomarker_1000.csv"), 
                        pd.read_csv("data/alzheimer_biomarker_2000.csv")]) \
                            .dropna(subset=["fulltext"]).reset_index(drop=True) \
                                .astype(str)
    return papers

@st.cache_data(show_spinner=False)
def load_paper(id):

    papers = load_literature()

    return papers[papers["pmcid"] == id]
    

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name, task):
    # START OF DOCUMENTATION CODE: https://huggingface.co/docs/transformers/en/model_doc/auto
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if task == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif task == "tokenclassify":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    # END OF CODE FROM https://huggingface.co/docs/transformers/en/model_doc/auto

    return model, tokenizer

summary_model, summary_tokenizer =  load_model_and_tokenizer("facebook/bart-large-cnn", "seq2seq")
ner_model, ner_tokenizer = load_model_and_tokenizer("alvaroalon2/biobert_genetic_ner", "tokenclassify")

@st.cache_data()
def generate_summary(text, max_length=2000, min_length=50):

    inputs = summary_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

@st.cache_data()
def perform_ner(text):

    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = ner_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [ner_model.config.id2label[t.item()] for t in predictions[0]]
    
    entities = []
    current_entity = ""
    current_label = ""
    
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity.strip(), current_label))
            current_entity = token.replace("#", "")
            current_label = label[2:]
        elif label.startswith("I-"):
            current_entity += " " + token.replace("#", "")
        else:
            if current_entity:
                entities.append((current_entity.strip(), current_label))
            current_entity = ""
            current_label = ""
    
    if current_entity:
        entities.append((current_entity.strip(), current_label))

    for i, (e, l) in enumerate(entities):
        entities[i] = (e.replace(" ", ""), l)

    
    return entities