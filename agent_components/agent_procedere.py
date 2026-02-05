from rag_components.retrieval import retrieve
from langchain_core.messages import SystemMessage
from rag.query import clean_docs2


def get_agent_output(chat_history, global_doc_list, llm): 
    clean_docs = clean_docs2(global_doc_list)
    message = f""" 
                You are a chemical industry process expert and natural language query formulation.
                You are given context and information from a chat histroy.
                It includes important context from reports, user questions and domain related citations.
                Based on this ONLY create a query for RAG in natural language and ONLY output the query.
                Do not add anything else. Don't blab. Don't explain. Don't argue. Don't add any text other than the query.
                Also use the following context to create a better query: {clean_docs}.
                Even with little to no context force an output that resembles a query for RAG.
                Always follow these instructions.
                NEVER ASK FOR MORE INFORMATION OR MORE CONTEXT.
                NEVER THINKG THERE IS TOO LITTLE INFORMATION.
                NEVER PROMPT THE USER FOR FURTHER INFORMATION OR CONTEXT.
                ALWAYS OUTPUT A RAG QUERY.
                Example: rectification usual problems, distillation common mishaps
                Always only output one query.
                Make sure the query you output matches the contents of a database with books, scientific papers, documents.
                Queires should be short to land in the best space for retrieval.     
    """
    message = "".join(message)
    system_prompt = SystemMessage(content=message)
    # print("Chat history in procedere agent: ", chat_history)
    prompt = [system_prompt] + chat_history
    print("procedere Prompt: ", prompt)
    query = llm.invoke(prompt)
    #docs = retrieve(query, 3)
    # then here we could format this further.
    #response = "Here are some relevant documents from the procedere manuals." # = docs
    print("Procedere Agent return: ", query)
    docs = query
    return docs

# NOTE: OOOOOOOR:
# query = query = llm.invoke(system_prompt, chat_history)
# global_chat_list.append(query)
# actually, do not append the query to the global chat list because it is not needed downstream.
# the queries are only relevant to the agents
# global_chat_list.append(response)


