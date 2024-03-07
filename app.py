import streamlit as st 
from streamlit_chat import message
from helper import get_qa_chain, create_vector_db

st.set_page_config(layout="wide",page_title="Chat with PDF")


def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = get_qa_chain()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    st.header("Chat with your PDF")
    create_embeddings = st.button("Create Embeddings")

    if create_embeddings:
        with st.spinner('Embeddings are in process...'):
            create_vector_db()
        st.success('Embeddings are created successfully!')

    st.subheader("Chat Here")
    user_input = st.text_input("",key="input")

    #initialize session state for generted response and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am an AI assitance how can I help?"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
    # Search the database for a response based on user input and update session state
    if user_input:
        answer = process_answer({'query': user_input})
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

        

if __name__ == "__main__":
    main()