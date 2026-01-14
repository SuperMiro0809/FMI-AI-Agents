import streamlit as st
import chromadb
import ollama

st.set_page_config(page_title="RAG Prompt Engineering")
st.title("Prompt Engineering RAG (Ollama)")

client = chromadb.PersistentClient()
col = client.get_collection("peg")

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Въпрос за Prompt Engineering")
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    emb = ollama.embeddings(
        model="llama3.2:latest",
        prompt=q
    )["embedding"]

    res = col.query(
        query_embeddings=[emb],
        n_results=5
    )

    context = "\n\n".join(res["documents"][0])

    answer = ollama.chat(
        model="llama3.2:latest",
        messages=[{
            "role": "user",
            "content": f"""Use only this context:

{context}

Question: {q}
"""
        }]
    )["message"]["content"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
