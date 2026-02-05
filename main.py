from agent_components import agent_chemical, agent_procedere, agent_summary, agent_ts
from src.llm_backend import get_local_llm
from langchain_core.documents import Document


# llm = "Dummy LLM"

import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import format_document

# LangChain message classes (optional but explicit)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.config import load_config

def main() -> None:
    context = ""
    cfg = load_config()

    global_chat_list = []
    global_doc_list = []
    llm = get_local_llm(cfg)

    # time_series_data_example = AIMessage(content="time series data", name="time series data")
   
    ### 
    # #we might not even need a time series agent. 
    # # if anything it just could describe it a bit better but yeah
    # response_ts = agent_ts.get_agent_output(global_chat_list, llm)
    # global_chat_list.append(AIMessage(content=response_ts, name = "agent_ts"))
    ### 

    ### MAKING INPUT FOR AGENTS: USER PROMPT + DATA

    time_series_data_example = Document(page_content="Here is a report from a time series.", metadata={"source": "https://example.com", "page": 1, "chunk": 1, "section": 1})
    global_doc_list.append(time_series_data_example)
    alarm_data_example = Document(page_content="Here is some additional data from alarm logs.", metadata={"source": "https://example.com", "page": 1, "chunk": 1, "section": 1})
    global_doc_list.append(alarm_data_example)
    user_prompt = HumanMessage(content=f"""Help me analyse the problem.""") # append user prompt later because we dont need user prompt for getting this type of data. We input give it everything we can manually and then let it work.
    global_chat_list.append(user_prompt)

    doc_prompt = PromptTemplate.from_template(
    "Source: {source}\n"
    "Page/Section: {page}{section}\n"
    "Chunk: {chunk}\n"
    "{page_content}"
    )

    formatted_docs = []
    for d in global_doc_list:
        # If some metadata keys are missing (e.g., page/section/chunk), you can normalize upstream
        formatted_docs.append(format_document(d, doc_prompt))

    context_block = "\n\n---\n\n".join(formatted_docs)
    print("Context: ", context_block)

    # INPUT: User prompt for perfect query
    # INPUT2: All the data we can give it (manually) or the data we want it to analyse (from before)
    response_procedere = agent_procedere.get_agent_output(global_chat_list, context_block, llm)
    global_chat_list.append(response_procedere)

    # # INPUT: User prompt for perfect query
    # # INPUT2: All the data we can give it (manually) or the data we want it to analyse (from before)
    # response_chemical = agent_chemical.get_agent_output(global_chat_list, llm)
    # global_chat_list.append(AIMessage(content=response_chemical, name = "agent_chemical"))



    # response = agent_summary.get_agent_output(global_chat_list, global_doc_list, llm) # <--- this already returns an AIMessage. Fix for every one above
    # # global_chat_list.append(AIMessage(content=response, name = "agent_summary"))
    # global_chat_list.append(response)

    for x in global_chat_list:
        print(x)

if __name__ == "__main__":
    main()