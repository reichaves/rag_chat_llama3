# RAG Conversacional com Upload de PDF e Histórico de Chat

Este projeto implementa um sistema de Recuperação de Informações Aumentada por Geração (RAG) conversacional usando Streamlit, LangChain, e modelos de linguagem de grande escala. O aplicativo permite que os usuários façam upload de documentos PDF, façam perguntas sobre o conteúdo desses documentos, e mantenham um histórico de chat para contexto em conversas contínuas.

## Autor

Reinaldo Chaves (reichaves@gmail.com)

VERSÃO ON-LINE - VEJA O SITE **[AQUI]([https://chatbotdepdfs.streamlit.app/](https://rag-chat-gemma2.streamlit.app/)**

## Características

- Interface de usuário Streamlit com tema dark
- Upload de múltiplos arquivos PDF
- Processamento de documentos usando LangChain e ChromaDB
- Geração de respostas usando o modelo Gemma2-9b-It da Groq
- Embeddings de texto usando o modelo all-MiniLM-L6-v2 do Hugging Face
- Histórico de chat para manter o contexto da conversa
- Barra lateral com orientações importantes para o usuário

## Requisitos

- Python 3.7+
- Streamlit
- LangChain
- ChromaDB
- PyPDF2
- Transformers
- Outras dependências listadas em `requirements.txt`

## Instalação

1. Clone este repositório:
   ```
   git clone https://github.com/seu_usuario/seu_repositorio.git
   cd seu_repositorio
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Configure as variáveis de ambiente ou tenha em mãos:
   - Chave da API Groq
   - Token da API Hugging Face

## Uso

1. Execute o aplicativo Streamlit:
   ```
   streamlit run app.py
   ```

2. Abra o navegador e acesse o endereço local mostrado no terminal.

3. Insira suas chaves de API quando solicitado.

4. Faça upload de um ou mais arquivos PDF.

5. Faça perguntas sobre o conteúdo dos documentos na caixa de entrada de texto.

## Como funciona

1. **Upload de Documentos**: Os usuários fazem upload de arquivos PDF, que são processados e divididos em chunks menores.

2. **Criação de Embeddings**: O texto é convertido em embeddings usando o modelo Hugging Face.

3. **Armazenamento de Vetores**: Os embeddings são armazenados em um banco de dados ChromaDB para recuperação eficiente.

4. **Processamento de Perguntas**: As perguntas dos usuários são contextualizadas com base no histórico do chat.

5. **Recuperação de Informações**: O sistema recupera os chunks de texto mais relevantes com base na pergunta.

6. **Geração de Respostas**: O modelo Gemma2-9b-It da Groq gera uma resposta com base nos chunks recuperados e na pergunta.

7. **Manutenção do Histórico**: O histórico do chat é mantido para fornecer contexto em conversas contínuas.

## Avisos Importantes

- Não compartilhe documentos contendo informações sensíveis ou confidenciais.
- As respostas geradas pela IA podem conter erros ou imprecisões. Sempre verifique as informações importantes.
- Este projeto é para fins educacionais e de demonstração. Use com responsabilidade.

## Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue para discutir mudanças importantes antes de fazer um pull request.

## Licença

[MIT License](LICENSE)
