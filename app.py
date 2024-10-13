import streamlit as st


from auth import retrieve_similar_articles


st.set_page_config("Literature Review", "💡")


def main():

    st.header("Literature Review")
    st.subheader("Literatures avilable for: Alzheimer Disease Biomarkers and related content")

    keyword = st.time_input("Input keyword for research")

    st.write(keyword)