from agent_components import agent_chemical, agent_procedere, agent_summary, agent_ts
from src.llm_backend import get_local_llm
from langchain_core.documents import Document


# llm = "Dummy LLM"

import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp

# LangChain message classes (optional but explicit)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.config import load_config

def main() -> None:
    context = ""
    cfg = load_config()

    global_chat_list = []
    global_doc_list = []
    llm = get_local_llm(cfg)

    system_message = SystemMessage(content=f"""You are a helpful assistant. Answer the user's questions. If available, use the follwing context: {global_doc_list}""")
    user_prompt = HumanMessage(content="What do you see in these messages?")
    global_chat_list.append(system_message)
    global_chat_list.append(user_prompt)

    # time_series_data_example = AIMessage(content="time series data", name="time series data")
    time_series_data_example = Document(page_content="time series data", metadata={"source": "https://example.com"})
    global_doc_list.append(time_series_data_example)

    #we might not even need a time series agent. 
    # if anything it just could describe it a bit better but yeah
    response_ts = agent_ts.get_agent_output(global_chat_list, llm)
    global_chat_list.append(AIMessage(content=response_ts, name = "agent_ts"))

    response_procedere = agent_procedere.get_agent_output(global_chat_list, llm)
    global_chat_list.append(AIMessage(content=response_procedere, name = "agent_procedere"))

    response_chemical = agent_chemical.get_agent_output(global_chat_list, llm)
    global_chat_list.append(AIMessage(content=response_chemical, name = "agent_chemical"))



    response = agent_summary.get_agent_output(global_chat_list, global_doc_list, llm) # <--- this already returns an AIMessage. Fix for every one above
    # global_chat_list.append(AIMessage(content=response, name = "agent_summary"))
    global_chat_list.append(response)

    for x in global_chat_list:
        print(x)

if __name__ == "__main__":
    main()