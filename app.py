"""
app.py

A Flask web app that:
1) Loads the vectorstore from db_store/ (which is created by analyzer.py).
2) Provides a chat interface with RAG (local docs) + optional web fallback.
3) Stacks messages vertically in a single column (one under the other).
4) Displays the Faten logo (place faten_logo.png in static/).

Run:
  python app.py
Then open http://localhost:5000
"""

import os
import json

from flask import Flask, render_template, request, jsonify

###############################################################################
# ENVIRONMENT SETUP
###############################################################################

os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

###############################################################################
# LLM
###############################################################################
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

###############################################################################
# LOAD VECTORSTORE FROM db_store
###############################################################################
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document as LangchainDocument
from langchain.schema import Document

# We assume you have already run analyzer.py to build the vectorstore
PERSIST_DIRECTORY = "db_store"

# If there's no data in db_store, you can show an error or gracefully handle it
if not (os.path.isdir(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY)):
    print(f"WARNING: No vectorstore found in '{PERSIST_DIRECTORY}'. Please run analyzer.py first.")
    # We won't raise an error; but some queries might fail if there's no data.

from langchain_nomic.embeddings import NomicEmbeddings

embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

def load_vectorstore():
    if os.path.isdir(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("Loading existing vector database from:", PERSIST_DIRECTORY)
        vectordb = Chroma(
            collection_name="my_collection",
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
        )
        return vectordb
    else:
        print(f"No data found in {PERSIST_DIRECTORY}. Returning None.")
        return None

###############################################################################
# WEB SEARCH TOOL (OPTIONAL)
###############################################################################
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

###############################################################################
# PROMPTS & LOGIC (RAG)
###############################################################################

router_instructions = """You are an expert at deciding if a user question can be answered from the local database or if web search is needed.

Return JSON like:
{{
  "datasource": "vectorstore" or "websearch"
}}"""

doc_grader_instructions = """You are a grader. If doc is relevant to question => yes; else no."""

doc_grader_prompt = """DOC: {document}
QUESTION: {question}

Return JSON:
{{
  "binary_score": "yes" or "no"
}}
"""

rag_prompt = """You are an assistant for question-answering tasks.

Context:
{context}

Question:
{question}

Provide a concise answer in up to three sentences.
Answer:
"""

hallucination_grader_instructions = """Return only yes/no if the answer is grounded in the given FACTS, no explanation."""

hallucination_grader_prompt = """FACTS:
{documents}

STUDENT ANSWER: {generation}

Return JSON:
{{
  "binary_score": "yes" or "no"
}}
"""

answer_grader_instructions = """Return only yes/no if the answer addresses the question."""

answer_grader_prompt = """QUESTION: {question}
STUDENT ANSWER: {generation}

Return JSON:
{{
  "binary_score": "yes" or "no"
}}
"""

def route_question(question: str) -> str:
    route_resp = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions),
         HumanMessage(content=question)]
    )
    try:
        source = json.loads(route_resp.content).get("datasource", "vectorstore")
        if source not in ["websearch", "vectorstore"]:
            source = "vectorstore"
    except:
        source = "vectorstore"
    return source

def grade_documents(question: str, docs):
    filtered_docs = []
    web_search_flag = False
    for d in docs:
        prompt = doc_grader_prompt.format(document=d.page_content, question=question)
        resp = llm_json_mode.invoke([
            HumanMessage(content=prompt),
            SystemMessage(content=doc_grader_instructions)
        ])
        try:
            score = json.loads(resp.content)["binary_score"].lower()
            if score == "yes":
                filtered_docs.append(d)
            else:
                web_search_flag = True
        except:
            web_search_flag = True
    return filtered_docs, web_search_flag

def generate_answer(question: str, docs):
    context_text = "\n\n".join([d.page_content for d in docs])
    prompt = rag_prompt.format(context=context_text, question=question)
    generation = llm.invoke([HumanMessage(content=prompt)])
    return generation.content

def check_hallucination(docs, answer: str):
    facts = "\n\n".join([d.page_content for d in docs])
    prompt = hallucination_grader_prompt.format(documents=facts, generation=answer)
    resp = llm_json_mode.invoke([
        SystemMessage(content=hallucination_grader_instructions),
        HumanMessage(content=prompt)
    ])
    try:
        return json.loads(resp.content)["binary_score"].lower()
    except:
        return "no"

def check_answer_relevance(question: str, answer: str):
    prompt = answer_grader_prompt.format(question=question, generation=answer)
    resp = llm_json_mode.invoke([
        SystemMessage(content=answer_grader_instructions),
        HumanMessage(content=prompt)
    ])
    try:
        return json.loads(resp.content)["binary_score"].lower()
    except:
        return "no"

def faten(question: str, vectordb: Chroma, max_retries=3):
    """
    1) Route question
    2) Retrieve docs from local DB
    3) If none relevant => fallback to web
    4) Generate answer
    5) Check hallucination & correctness => up to max_retries
    """
    datasource = route_question(question)

    # Retrieve from local DB
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    # Grade docs
    relevant_docs, need_web = grade_documents(question, docs)
    if len(relevant_docs) == 0:
        need_web = True
    docs = relevant_docs

    # Fallback if needed
    if need_web and len(docs) == 0:
        search_res = web_search_tool.invoke({"query": question})
        if isinstance(search_res, list):
            combined = "\n".join([r["content"] for r in search_res])
        else:
            combined = str(search_res)
        docs = [Document(page_content=combined)]

    # Generate answer
    answer = ""
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        answer = generate_answer(question, docs)

        hall_score = check_hallucination(docs, answer)
        if hall_score == "yes":
            ans_score = check_answer_relevance(question, answer)
            if ans_score == "yes":
                break
            else:
                continue
        else:
            continue

    return answer

###############################################################################
# FLASK APP
###############################################################################

app = Flask(__name__, static_folder="static")

# Load vectorstore created by analyzer.py
vectordb = load_vectorstore()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    if not question.strip():
        return jsonify({"answer": "Please enter a question."})

    if not vectordb:
        # If no data found, just respond with an error or a default message
        return jsonify({"answer": "No local data found. Please run analyzer.py first."})

    final_answer = faten(question, vectordb)
    return jsonify({"answer": final_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
