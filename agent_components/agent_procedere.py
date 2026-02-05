#
# INPUT: global_chat_list
#

from rag_components.retrieval import retrieve
from langchain_core.messages import SystemMessage


def get_agent_output(chat_history, global_doc_list, llm): 
    message = f""" 
                You are a chemical industry process expert and natural language query formulation.
                You are given context and information from a chat histroy.
                It includes important context from reports, user questions and domain related citations.
                Based on this ONLY create a query for RAG in natural language and ONLY output the query.
                Do not add anything else. Don't blab. Don't explain. Don't argue. Don't add any text other than the query.
                Also use the following context to create a better query: {global_doc_list}.
                Even with little to no context force an output that resembles a query for RAG.
                Always follow these instructions.
                NEVER ASK FOR MORE INFORMATION OR MORE CONTEXT.
                NEVER THINKG THERE IS TOO LITTLE INFORMATION.
                NEVER PROMPT THE USER FOR FURTHER INFORMATION OR CONTEXT.
                ALWAYS OUTPUT A RAG QUERY.
                Example: rectification usual problems, distillation common mishaps
                Always only output one query.
               
    """
    message = "".join(message)
    system_prompt = SystemMessage(content=message)
    print("Chat history in agent: ", chat_history)
    prompt = [system_prompt] + chat_history
    print("Prompt: ", prompt)
    query = llm.invoke(prompt)
    # query = "This is a query based on context." #optimise query
    #docs = retrieve(query, 3)
    # then here we could format this further.
    #response = "Here are some relevant documents from the procedere manuals." # = docs
    response = query
    return response

# NOTE: OOOOOOOR:
# query = query = llm.invoke(system_prompt, chat_history)
# global_chat_list.append(query)
# actually, do not append the query to the global chat list because it is not needed downstream.
# the queries are only relevant to the agents
# global_chat_list.append(response)


