#
# INPUT: global_chat_list
#

chat_history = ""
# NOTE: OR:
# get(global_chat_history)
system_prompt = """ 
                You are a time series analysis specialist and an expert in chemical distillation science. 
                You are given context and information from a chat histroy. 
                It includes important context from reports, user questions and domain related citations.
                Based on a report of a distillation time series you need to collect sources that help explain a time series.
                You are given a time series report that describes metrics from a distillation column precedure.
                Based on this create a actionable and insightful summary of the given contents.
                Include any anomalies or things that may not be obvious from the raw numbers alone.
                """

def get_agent_output(chat_history, llm): 
    #response = llm.invoke(system_prompt, chat_history)
    response = "Some analysis about time series."
    return response

# NOTE: OOOOOOOR:
# query = query = llm.invoke(system_prompt, chat_history)
# global_chat_list.append(query)
# actually, do not append the query to the global chat list because it is not needed downstream.
# the queries are only relevant to the agents
#global_chat_list.append(response)

