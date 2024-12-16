
import os
import time
from uuid import uuid4
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from jira import JIRA
from langchain_core.runnables import RunnablePassthrough
import easyocr
from PIL import Image

# Load environment variables
load_dotenv()

# Helper function for formatting Jira documents
def format_jira_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Pinecone and Jira configurations
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_USER_EMAIL = os.getenv("JIRA_USER_EMAIL")
PROJECT_KEY = os.getenv("PROJECT_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "embedding-new"
embedding_dimension = 4096

# Ensure Pinecone index exists
if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Jira issue creation
def create_jira_issue(summary, description):
    jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USER_EMAIL, JIRA_API_TOKEN))
    issue = jira.create_issue(
        project=PROJECT_KEY,
        summary=summary,
        description=description,
        issuetype={'name': 'Task'},
    )
    return issue

# Functions for modes
def suggest_code(task_description, language):
    model = ChatOllama(model="llama3")
    prompt_template = f"""
    You are a helpful AI assistant specializing in {language} programming.
    Based on the user's request, suggest a {language} code snippet.

    Task: {{task_description}}

    Suggested Code:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    full_prompt = prompt.format(task_description=task_description)

    response = model.invoke(full_prompt)
    return response.content


def review_code(code_snippet, language):
    model = ChatOllama(model="llama3")
    prompt_template = f"""
    You are a code reviewer AI specializing in {language} programming. Review the following code for correctness, 
    optimization, and best practices. Provide feedback.

    Code:
    {{code_snippet}}

    Feedback:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    full_prompt = prompt.format(code_snippet=code_snippet)

    response = model.invoke(full_prompt)
    return response.content


def design_to_code(image_path, language):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    extracted_text = "\n".join([text[1] for text in result])

    model = ChatOllama(model="llama3")
    prompt_template = f"""
    You are an AI assistant specializing in converting designs into {language} programming logic. 
    Based on the extracted text and components of the diagram, generate the corresponding code.

    Extracted Diagram Content:
    {{extracted_text}}

    Suggested Code:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    full_prompt = prompt.format(extracted_text=extracted_text)

    response = model.invoke(full_prompt)
    return response.content

# Streamlit app
def main():
    st.title("Multi-Purpose AI Assistant")
    st.sidebar.title("Select Mode")
    mode = st.sidebar.radio("Choose an action", ("Suggest Code", "Review Code", "Design to Code", "Issue Management in Jira"))

    if mode == "Issue Management in Jira":
        st.subheader("Jira ticket creation")

        if "file_processed" not in st.session_state:
            st.session_state.file_processed = False

        uploaded_file = st.file_uploader("Upload a text file", type="txt")
        if uploaded_file is not None and not st.session_state.file_processed:
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content", content, height=300)

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            document = [Document(page_content=content, metadata={"source": uploaded_file.name})]
            chunks = text_splitter.split_documents(document)

            embeddings_model = OllamaEmbeddings(model="llama3")
            uuids = [str(uuid4()) for _ in chunks]

            for chunk, uuid in zip(chunks, uuids):
                embedding = embeddings_model.embed_query(chunk.page_content)
                embedding = np.array(embedding, dtype=np.float32)
                index.upsert(vectors=[(uuid, embedding, {"text": chunk.page_content, **chunk.metadata})])

            st.success("File content successfully embedded and ingested into Pinecone.")
            st.session_state.file_processed = True

        query = st.text_input("Enter your query:")
        if query:
            llm = ChatOllama(model="llama3")
            embeddings_model = OllamaEmbeddings(model="llama3")
            vectorstore = PineconeVectorStore(index=index, embedding=embeddings_model)

            jira_template = """You are a system designed to fetch and summarize error-related information.

            Use the provided context to identify and extract the following:
            1. Issue Summary: Provide a concise summary of the issue.
            2. Issue Details: Provide a brief description or explanation of the issue.

            Give response starting with Isuue Summary followed by Issue Details don't include any special characters at starting and ending.

            Context:
            {context}

            Question: {question}

            Response:"""

            custom_jira_prompt = PromptTemplate.from_template(jira_template)

            jira_chain = (
                {"context": vectorstore.as_retriever() | format_jira_docs, "question": RunnablePassthrough()}
                | custom_jira_prompt
                | llm
            )

            res = jira_chain.invoke(query)
            llm_response = res.content.strip()
            st.subheader("LLM Response")
            st.write(llm_response)

            lines = llm_response.split("\n")
            issue_summary = lines[0].strip() if len(lines) > 0 else "No summary provided"
            issue_details = "\n".join(lines[1:]).strip() if len(lines) > 1 else "No details provided"

            if st.button("Create Jira Issue"):
                jira_issue = create_jira_issue(issue_summary, issue_details)
                st.success(f"Jira issue created: {jira_issue.key}")

    elif mode == "Suggest Code":
        st.subheader("Code Suggestion")
        task_description = st.text_area("Enter a task description:")
        language = st.sidebar.selectbox("Programming Language", ["Python", "Java", "C++", "JavaScript", "Go"])

        if st.button("Generate Code"):
            if task_description.strip():
                with st.spinner("Generating code..."):
                    code = suggest_code(task_description, language)
                st.code(code, language=language.lower())
            else:
                st.error("Please enter a valid task description.")

    elif mode == "Review Code":
        st.subheader("Code Review")
        code_snippet = st.text_area("Paste your code snippet:")
        language = st.sidebar.selectbox("Programming Language", ["Python", "Java", "C++", "JavaScript", "Go"])

        if st.button("Review Code"):
            if code_snippet.strip():
                with st.spinner("Reviewing code..."):
                    feedback = review_code(code_snippet, language)
                st.write("### Feedback")
                st.text(feedback)
            else:
                st.error("Please enter a valid code snippet.")

    elif mode == "Design to Code":
        st.subheader("Design to Code Mapping")
        uploaded_file = st.file_uploader("Upload a diagram (e.g., flowchart or logic diagram)", type=["png", "jpg", "jpeg"])
        language = st.sidebar.selectbox("Programming Language", ["Python", "Java", "C++", "JavaScript", "Go"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Diagram", use_column_width=True)
            if st.button("Convert to Code"):
                with st.spinner("Processing diagram..."):
                    with open("temp_image.jpg", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    code = design_to_code("temp_image.jpg", language)
                st.subheader("Generated Code")
                st.code(code, language=language.lower())


if __name__ == "__main__":
    main()

