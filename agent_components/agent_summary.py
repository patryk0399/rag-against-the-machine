from rag_components.retrieval import retrieve
from langchain_core.messages import SystemMessage
from rag.query import clean_docs2


def get_agent_output(chat_history, global_doc_list, llm): 
    clean_docs = clean_docs2(global_doc_list)
    message = f""" 
                You are an expert summaries.
                You are given context and information from a chat histroy.
                It includes important context from reports, user questions and domain related citations.
                Based on this create a actionable and insightful summary of the given contents. 
                Additional context from documents: {clean_docs}   
    """
    message = "".join(message)
    system_prompt = SystemMessage(content=message)
    # print("Chat history in procedere agent: ", chat_history)
    prompt = [system_prompt] + chat_history
    print("procedere Prompt: ", prompt)
    summary = llm.invoke(prompt)
    #docs = retrieve(query, 3)
    # then here we could format this further.
    #response = "Here are some relevant documents from the procedere manuals." # = docs
    print("Summary Agent return: ", summary)
    return summary

# NOTE: OOOOOOOR:
# query = query = llm.invoke(system_prompt, chat_history)
# global_chat_list.append(query)
# actually, do not append the query to the global chat list because it is not needed downstream.
# the queries are only relevant to the agents
# global_chat_list.append(response)



