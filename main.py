"""
main.py

A local AI chat system that uses:
1) A local LLM (via ChatOllama)
2) A local vectorstore of documents
3) Optional fallback to web search (TavilySearch) if the vectorstore doesn't have relevant info

Steps:
- Route user question (vectorstore vs. web search)
- Retrieve documents
- Grade documents for relevance
- Generate an answer (RAG)
- Check for hallucination
- Check if the answer addresses the question
- Return a final answer to the user

Author: You
"""

import os
import getpass
import json

# 1) LLM: Local Large Language Model
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

###############################################################################
# ENVIRONMENT SETUP
###############################################################################

def _set_env(var: str):
    """
    If environment variable var is not set, prompt the user to input it.
    This is used to ensure TAVILY_API_KEY (for web search) is set.
    """
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# If you intend to use TavilySearch, ensure TAVILY_API_KEY is set
_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

###############################################################################
# INITIALIZE LLM
###############################################################################

# Path/name of local LLM; update this to the model you have installed for ChatOllama
local_llm = "llama3.2:3b-instruct-fp16"

# Normal text LLM (returns text)
llm = ChatOllama(model=local_llm, temperature=0)

# JSON-mode LLM (returns well-formed JSON)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

###############################################################################
# VECTORSTORE FROM LOCAL FILES
###############################################################################

"""
You'll need to install or update to `langchain_community` for DirectoryLoader if using:
  from langchain_community.document_loaders import DirectoryLoader

If you still have the older imports, you might see a deprecation warning. 
Below is the recommended approach using `langchain_community.document_loaders.DirectoryLoader`.
"""

try:
    from langchain_community.document_loaders import DirectoryLoader
except ImportError:
    print("Failed to import DirectoryLoader from langchain_community. Install or update 'langchain_community' package.")
    raise

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

# -- Load documents from your local folder
#    Replace "path/to/your/folder" with the correct path to your documents
docs_loader = DirectoryLoader("C:/Users/asus/Desktop/MVPFATEN/Info", glob="*.txt")
docs_list = docs_loader.load()

# -- Split documents for embedding
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

# -- Create vectorstore
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)

# -- Create retriever
retriever = vectorstore.as_retriever(k=3)

###############################################################################
# WEB SEARCH TOOL (OPTIONAL)
###############################################################################

from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

###############################################################################
# PROMPTS / INSTRUCTIONS
###############################################################################

# A) Router instructions
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains local documents about various topics.

Use the vectorstore for questions that might relate to the knowledge in these local documents.
For all else, especially current or external events, use websearch.

Return JSON with a single key, 'datasource', that is 'websearch' or 'vectorstore' depending on the question.
"""

# B) Document relevance grader
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
"""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

Carefully and objectively assess whether the document contains at least some information relevant to the question.

Return JSON with a single key, 'binary_score', with value 'yes' or 'no' to indicate relevance.
"""

# C) RAG prompt
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this question using only the above context. 

Use three sentences maximum, and keep the answer concise.

Answer:
"""

# D) Hallucination grader
hallucination_grader_instructions = """
You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER.

Grade criteria:
(1) The STUDENT ANSWER must be grounded in the FACTS. 
(2) The STUDENT ANSWER must not contain "hallucinated" information outside the scope of the FACTS.

Return JSON with:
  "binary_score": "yes" or "no"
  "explanation": <step-by-step explanation of why>
"""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys:
  "binary_score": "yes" or "no"
  "explanation": <your reasoning here>
"""

# E) Answer grader
answer_grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Criteria:
(1) The STUDENT ANSWER helps answer the QUESTION.

Return JSON with:
  "binary_score": "yes" or "no"
  "explanation": <step-by-step explanation of why>
"""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys:
  "binary_score": "yes" or "no"
  "explanation": <your reasoning here>
"""

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def format_docs(docs):
    """
    Concatenate the text from a list of documents with spacing in between.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def route_question(question: str) -> str:
    """
    Route a user question to either 'vectorstore' or 'websearch' based on LLM classification.
    """
    route_resp = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions),
         HumanMessage(content=question)]
    )
    try:
        source = json.loads(route_resp.content).get("datasource", "vectorstore")
        if source in ["websearch", "vectorstore"]:
            return source
    except:
        pass
    return "vectorstore"  # fallback if parsing fails

def retrieve_from_vectorstore(question: str):
    """
    Retrieve top documents from the local vectorstore for the given question.
    """
    docs = retriever.invoke(question)
    return docs

def grade_documents(question: str, docs):
    """
    Grade each document for relevance to the question.
    Return the subset that is relevant and a flag indicating if we should do websearch.
    """
    filtered_docs = []
    web_search_flag = False
    
    for d in docs:
        prompt = doc_grader_prompt.format(document=d.page_content, question=question)
        grade_resp = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions),
             HumanMessage(content=prompt)]
        )
        try:
            score = json.loads(grade_resp.content).get("binary_score", "").lower()
            if score == "yes":
                filtered_docs.append(d)
            else:
                web_search_flag = True
        except:
            # If parsing fails, assume 'not relevant'
            web_search_flag = True

    return filtered_docs, web_search_flag

def do_web_search(question: str):
    """
    Perform a web search for the question and return results as a single Document.
    Includes debug printing to see what is returned by the search tool.
    """
    search_results = web_search_tool.invoke({"query": question})
    
    # Debug print to see what search_results looks like
    print("DEBUG: search_results =", search_results)
    
    # If the result is not a list of dicts, handle it gracefully
    if not isinstance(search_results, list):
        print("ERROR: Web search returned a non-list. Possibly an error string or invalid API key?")
        from langchain.schema import Document
        web_doc = Document(page_content=str(search_results))
        return web_doc
    
    # Now, we assume search_results is a list of dict with "content" key
    combined_content = "\n".join([r["content"] for r in search_results])
    from langchain.schema import Document
    web_doc = Document(page_content=combined_content)
    return web_doc

def generate_answer(question: str, docs):
    """
    Use RAG prompt with the docs to generate an answer to the question.
    """
    docs_text = format_docs(docs)
    prompt = rag_prompt.format(context=docs_text, question=question)
    generation = llm.invoke([HumanMessage(content=prompt)])
    return generation.content

def check_hallucination(docs, generated_answer: str):
    """
    Check if the generated answer is grounded in the docs.
    Returns:
      - score: 'yes' if grounded, 'no' if not
      - explanation: text explanation
    """
    docs_text = format_docs(docs)
    prompt = hallucination_grader_prompt.format(documents=docs_text, generation=generated_answer)
    resp = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions),
         HumanMessage(content=prompt)]
    )
    try:
        parsed = json.loads(resp.content)
        return parsed["binary_score"].lower(), parsed["explanation"]
    except:
        return "no", "Could not parse hallucination grader response."

def check_answer_relevance(question: str, generated_answer: str):
    """
    Check if the generated answer addresses the user's question.
    Returns:
      - score: 'yes' if it addresses the question, 'no' otherwise
      - explanation: text explanation
    """
    prompt = answer_grader_prompt.format(question=question, generation=generated_answer)
    resp = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions),
         HumanMessage(content=prompt)]
    )
    try:
        parsed = json.loads(resp.content)
        return parsed["binary_score"].lower(), parsed["explanation"]
    except:
        return "no", "Could not parse answer grader response."

###############################################################################
# MAIN PIPELINE FUNCTION (faten)
###############################################################################

def faten(question: str, max_retries: int = 3):
    """
    Main function that:
    1. Routes the question
    2. Retrieves documents (vectorstore) or does web search
    3. Grades documents
    4. Possibly does web search if needed
    5. Generates an answer
    6. Checks for hallucination
    7. Checks if it addresses the question
    8. Returns final answer or re-tries up to max_retries
    """
    
    # Step 1: Route question
    datasource = route_question(question)
    print(f"Routing to: {datasource}")
    
    # Step 2: Retrieve or Web search
    if datasource == "vectorstore":
        docs = retrieve_from_vectorstore(question)
    else:
        # websearch route
        web_doc = do_web_search(question)
        docs = [web_doc]

    # Step 3: If we used vectorstore, grade documents
    #         If they're not relevant, fallback to websearch
    if datasource == "vectorstore":
        docs, need_web = grade_documents(question, docs)
        if need_web or len(docs) == 0:
            # fallback: do web search
            print("No relevant docs found or partial relevance. Searching the web...")
            web_doc = do_web_search(question)
            docs.append(web_doc)

    # Steps 4 & 5: Generate answer, up to max_retries attempts
    answer = ""
    attempts = 0

    while attempts < max_retries:
        attempts += 1
        print(f"Generate attempt #{attempts}")
        answer = generate_answer(question, docs)

        # Step 6: Hallucination check
        halluc_score, halluc_explanation = check_hallucination(docs, answer)
        print(f"Hallucination check: {halluc_score}, explanation: {halluc_explanation}")

        if halluc_score == "yes":
            # Step 7: Relevance check
            ans_score, ans_explanation = check_answer_relevance(question, answer)
            print(f"Answer relevance check: {ans_score}, explanation: {ans_explanation}")

            if ans_score == "yes":
                # Good to go
                break
            else:
                # The answer did not address the question well
                # Retry if we still have attempts left
                continue
        else:
            # The answer is not grounded in docs; let's retry
            continue

    return answer

###############################################################################
# RUN THE SCRIPT
###############################################################################

if __name__ == "__main__":
    user_question = input("Enter your question: ")
    final_answer = faten(user_question)
    print("\n=== FINAL ANSWER ===")
    print(final_answer)
