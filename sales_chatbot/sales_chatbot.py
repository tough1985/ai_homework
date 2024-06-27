import gradio as gr

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_retrieval_qa import MultiRetrievalQAChain

loader = TextLoader("")

# 初始化房产销售数据库
def initialize_db():
    with open("real_estate_sales_data.txt") as f:
        real_estate_sales = f.read()
    text_splitter = CharacterTextSplitter(        
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local("real_estate_sales")

# 初始化AI销售数据库
def initialize_ai_sales_db():
    with open("ai_sales_data.txt") as f:
        real_estate_sales = f.read()
    text_splitter = CharacterTextSplitter(        
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )
    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local("ai_sales")

def initialize_multichain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    retriever_infos = [
        {
            "name": "AI销售",
            "description": "适用回答与AI销售相关的问题",
            "retriever": get_retrievel("ai_sales")
        },
        {
            "name": "房产销售",
            "description": "适用回答与房产销售相关的问题",
            "retriever": get_retrievel("real_estates_sales")
        },
    ]
    global multi_chain
    multi_chain = MultiRetrievalQAChain.from_retrievers(llm, retriever_infos)
    for item in multi_chain.destination_chains.items():
        item[1].return_source_documents = True

def get_retrievel(vector_store_dir):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.5})
    return retriever

# # 初始化聊天机器人
# def initialize_chatbot(vector_store_dir: str="real_estates_sales"):
#     db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
#     llm = ChatOpenAI(model_name="gpt-4", temperature=0)
#     global SALES_BOT
#     SALES_BOT = RetrievalQA.from_chain_type(llm,
#                                             retriever=db.as_retriever(search_type="similarity_score_threshold",
#                                                                      search_kwargs={"score_threshold": 0.5}))
#     SALES_BOT.return_source_documents = True

#     return SALES_BOT

# 启动gradio
def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产&AI 销售",
        chatbot=gr.Chatbot(height=600)
    )
    demo.launch(share=True, server_name="0.0.0.0")

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    ans = multi_chain.invoke({"input": message})
    # ans = multi_chain.invoke({"query": message, "input": message})
    print(ans)
    return ans["result"]
    # ans = SALES_BOT({"query": message})
    # if ans["source_documents"]:
    #     print(f"[result]{ans['result']}")
    #     print(f"[source_documents]{ans['source_documents']}")
    #     return ans['result']
    # else:
    #     return "这个问题我需要问问领导"

if __name__ == "__main__":
    # initialize_ai_sales_db()
    # initialize_chatbot("ai_sales")
    initialize_multichain()
    launch_gradio()
    # initialize_db()
    
    