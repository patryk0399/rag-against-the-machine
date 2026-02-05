from agent_components import agent_chemical, agent_procedere, agent_summary, agent_ts
from src.llm_backend import get_local_llm
from langchain_core.documents import Document
#from rag.query import clean_docs


# llm = "Dummy LLM"

import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import format_document

# LangChain message classes (optional but explicit)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.config import load_config
#from rag.query import clean_docs2

def main() -> None:
    cfg = load_config()

    global_chat_list = []
    global_doc_list = []
    llm_chat = get_local_llm(cfg, 0)
    llm_query = get_local_llm(cfg, 1)
   
    ### 
    # #we might not even need a time series agent. 
    # # if anything it just could describe it a bit better but yeah
    # response_ts = agent_ts.get_agent_output(global_chat_list, llm)
    # global_chat_list.append(AIMessage(content=response_ts, name = "agent_ts"))
    ### 

    ### MAKING INPUT FOR AGENTS: USER PROMPT + DATA

    time_series_data_example = Document(page_content="Here is a report from a time series.", metadata={"source": "dummy data", "page": 1, "chunk": 1, "section": 1})
    global_doc_list.append(time_series_data_example)
    alarm_data_example = Document(page_content="Here is some additional data from alarm logs.", metadata={"source": "dummy data", "page": 1, "chunk": 1, "section": 1})
    global_doc_list.append(alarm_data_example)
    user_prompt = HumanMessage(content=f"""Help me analyse the problem.""") # append user prompt later because we dont need user prompt for getting this type of data. We input give it everything we can manually and then let it work.
    global_chat_list.append(user_prompt)
    
    # for each agent:
    # INPUT: User prompt for perfect query
    # INPUT2: All the data we can give it (manually) or the data we want it to analyse (from before)
    response_procedere = agent_procedere.get_agent_output(global_chat_list, global_doc_list, llm_query)
    global_doc_list.append(Document(page_content = response_procedere.content, metadata={"source": "agent_procedere", "page": 1, "chunk": 1, "section": 1}))# <-- later replace "agent_procedere" with repsonse_procedere.metadata.source.....
 
    response_chemical = agent_chemical.get_agent_output(global_chat_list, global_doc_list, llm_query)
    global_doc_list.append(Document(page_content = response_chemical.content, metadata={"source": "agent_chemical", "page": 1, "chunk": 1, "section": 1}))

    response = agent_summary.get_agent_output(global_chat_list, global_doc_list, llm_chat) # <--- this already returns an AIMessage. Fix for every one above
    global_chat_list.append(response)

    print("Messages: ")
    for x in global_chat_list:
        print(x)
    print("Docs: ")
    for x in global_doc_list:
        print(x)

if __name__ == "__main__":
    main()