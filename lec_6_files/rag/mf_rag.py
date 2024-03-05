# https://medium.com/@kv742000/creating-a-python-server-for-document-q-a-using-langchain-31a123b67935
from flask import Flask, request, jsonify

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.output_parsers import RegexParser
from dotenv import load_dotenv

import streamlit as st 
import pandas as pd
import numpy as np
import pprint

import json
from pathlib import Path
from pprint import pprint

load_dotenv()

app = Flask(__name__)

# Load docs describing March Fitness
loader = DirectoryLoader(f'mf_rules', glob="./*.docx", loader_cls=Docx2txtLoader)
documents = loader.load()
chunk_size_value = 1000
chunk_overlap=100
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value, chunk_overlap=chunk_overlap,length_function=len)
texts = text_splitter.split_documents(documents)
docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())
docembeddings.save_local("llm_faiss_index")
docembeddings = FAISS.load_local("llm_faiss_index",OpenAIEmbeddings())

# import JSON with scoring values
# file_path='./activity_types.json'
# data = json.loads(Path(file_path).read_text())
# loader = JSONLoader(file_path='./activity_types.json', jq_schema=".", json_lines=False, text_content=False)
# documents = loader.load()

prompt_template = """Use the following pieces of context, that explain the rules to a fitness competition, to answer questions pertaining to rules and scoring. If you don't know the answer, just say that you don't know, don't try to make up an answer.

This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser
)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)

@st.cache_data
def getanswer(query):
    relevant_chunks = docembeddings.similarity_search_with_score(query,k=2)
    chunk_docs=[]
    for chunk in relevant_chunks:
        chunk_docs.append(chunk[0])
    results = chain({"input_documents": chunk_docs, "question": query})
    text_reference=""
    for i in range(len(results["input_documents"])):
        text_reference+=results["input_documents"][i].page_content
    output={"Answer":results["output_text"],"Reference":text_reference}
    return output

# Streamlit app
if __name__ == "__main__":
    st.title("March fitness rules chatbot")
    question = st.text_input("Enter a question about rules or scoring in March fitness:")
    st.write("MF admin chatbot says:", getanswer(question))