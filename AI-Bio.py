import os
from sec import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key

# ---- Streamlit UI Layout ----
st.set_page_config(page_title="LangChain Chatbot", layout="centered")

# ---- Sidebar ----
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider("Creativity Level (Temperature)", 0.0, 1.0, 0.8, 0.1)
st.sidebar.write("Increase for more creative answers.")

st.sidebar.markdown("## About")
st.sidebar.info(
    "This chatbot searches for information about a person, retrieves their birth date, and mentions significant events during that period."
)

# ---- Title & Input Section ----
st.markdown("<h1 style='text-align: center;'>LangChain AI Chatbot 🤖</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Ask about a person and discover interesting facts! 🔍</h5>", unsafe_allow_html=True)

input_text = st.text_input("🔍 Enter a person's name:", placeholder="Type a name...")

# ---- Prompt Templates ----
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about {name} in 25 words."
)

second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template="When was {person} born?"
)

third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 3 major events around {dob} in the world, each event in 20 words."
)

# ---- Memory Buffers ----
person_memory = ConversationBufferMemory(input_key="name", memory_key="chat_history")
dob_memory = ConversationBufferMemory(input_key="person", memory_key="chat_history")
desc1_memory = ConversationBufferMemory(input_key="dob", memory_key="desc_history")

# ---- OpenAI LLM Initialization ----
llm = OpenAI(temperature=temperature)

# ---- LLM Chains ----
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="person", memory=person_memory)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="dob", memory=dob_memory)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key="desc1", memory=desc1_memory)

# ---- Sequential Chain ----
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=["name"],
    output_variables=["person", "dob", "desc1"],
    verbose=True
)

# ---- Run Query When Button is Clicked ----
if st.button("🔍 Search"):
    if input_text:
        response = parent_chain({"name": input_text})
        
        # ---- Display Results ----
        st.subheader("📌 Results")
        
        with st.expander("🧑 Person Summary"):
            st.info(person_memory.buffer)

        with st.expander("📅 Date of Birth"):
            st.info(dob_memory.buffer)

        with st.expander("🌍 Major Events Around That Time"):
            st.info(desc1_memory.buffer)
    else:
        st.warning("⚠️ Please enter a name before searching.")

# ---- Footer ----
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Made with ❤️ using LangChain & Streamlit</p>",
    unsafe_allow_html=True
)
