# -*- coding: utf-8
# Reinaldo Chaves (reichaves@gmail.com)
# Este projeto implementa um sistema de Recuperação de Informações Aumentada por Geração (RAG) conversacional 
# usando Streamlit, LangChain, e modelos de linguagem de grande escala - para entrevistar PDFs
# Geração de respostas usando o modelo Gemma2-9b-It da Groq
# Embeddings de texto usando o modelo all-MiniLM-L6-v2 do Hugging Face
#

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os
import tempfile

# Configurar o tema para dark
st.set_page_config(page_title="RAG Q&A Conversacional", layout="wide", initial_sidebar_state="expanded", page_icon="🤖", menu_items=None)

# Aplicar o tema dark
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stButton>button {
        color: #4F8BF9;
        border-radius: 20px;
        height: 3em;
        width: 200px;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar com orientações
st.sidebar.title("Menu")
st.sidebar.markdown("""
* Se encontrar erros de processamento, reinicie com F5. Utilize arquivos .PDF com textos não digitalizados como imagens.
* Para recomeçar uma nova sessão pressione F5.

**Atenção:** Os documentos que você compartilhar com o modelo de IA generativa podem ser usados pelo LLM para treinar o sistema. Portanto, evite compartilhar documentos PDF que contenham:
1. Dados bancários e financeiros
2. Dados de sua própria empresa
3. Informações pessoais
4. Informações de propriedade intelectual
5. Conteúdos autorais

E não use IA para escrever um texto inteiro! O auxílio é melhor para gerar resumos, filtrar informações ou auxiliar a entender contextos - que depois devem ser checados. Inteligência Artificial comete erros (alucinações, viés, baixa qualidade, problemas éticos)!

Este projeto não se responsabiliza pelos conteúdos criados a partir deste site.

**Sobre este app**

Este aplicativo foi desenvolvido por Reinaldo Chaves. Para mais informações, contribuições e feedback, visite o [repositório do projeto no GitHub](https://github.com/reichaves/rag_chat_gemma2).
""")

st.title("RAG conversacional com upload em PDF e histórico de bate-papo")
st.write("Carregue PDFs e converse com o conteúdo deles - aqui é usado o modelo de LLM Gemma2-9b-It e a plataforma de embeddings é all-MiniLM-L6-v2")

# Solicitar as chaves de API
groq_api_key = st.text_input("Insira sua chave de API Groq:", type="password")
huggingface_api_token = st.text_input("Insira seu token de API Hugging Face:", type="password")

if groq_api_key and huggingface_api_token:
    # Configurar o token da API do Hugging Face
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

    # Inicializar o modelo de linguagem e embeddings
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Faça o upload de um ou mais arquivos PDF: ", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            documents.extend(docs)
            os.unlink(temp_file_path)  # Remove temporary file

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        st.success(f"Processed {len(splits)} document chunks.")

        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "Você é um assistente para tarefas de resposta a perguntas. Sempre coloque no final das respostas: 'Todas as informações devem ser checadas com a(s) fonte(s) original(ais)'. Responda em Português do Brasil a menos que seja pedido outro idioma"
            "Use os seguintes pedaços de contexto recuperado para responder "
            "à pergunta. Se você não sabe a resposta, diga que "
            "não sabe. Use no máximo três frases e mantenha a "
            "resposta concisa."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Sua pergunta:")
        if user_input:
            with st.spinner("Processando sua pergunta..."):
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
            st.write("Assistente:", response['answer'])
            
            with st.expander("Ver histórico do chat"):
                for message in session_history.messages:
                    st.write(f"**{message.type}:** {message.content}")
else:
    st.warning("Por favor, insira tanto a chave da API do Groq quanto o token da API do Hugging Face.")
