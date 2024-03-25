import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

import google.generativeai as genai
import googleapiclient.discovery
import pandas as pd
import json


with st.sidebar:
    google_api_key = st.text_input(
        "Google API Key", key="google_api_key", type="password"
    )
    "Get Google API key for Youtube and Gemini"
with st.sidebar:
    youtube_max_results = st.number_input(
        "Max results in Youtube", 1
    )
    "Top N results in Youtube search. suggest < 50"

# for openAi

# with st.sidebar:
#     openai_api_key = st.text_input(
#         "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
#     )
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 Chat with Youtube")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the Youtube. Input the query for Youtube, I will help you to summarize the trend"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="E.g. Top stocks"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check key
    if not google_api_key:
        st.info("Please add your Google API key to continue.")
        st.stop()

    # RAG: Get Youtube search result 
    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = google_api_key)

    # Request body
    youtube_query = prompt
    max_results = youtube_max_results
    youtube_request = youtube.search().list(
            part="id,snippet",
            type='video',
            q=youtube_query,
            videoDuration='short',
            videoDefinition='high',
            maxResults=max_results
    )
    # Request execution
    youtube_response = youtube_request.execute()

    # print(youtube_response)
    # json_string = json.dumps(youtube_response, indent=4, sort_keys=True)
    # print(json_string)

    
    # save in dataframe, so we can look at it as the source
    video_info = {
        'videoId':[],
        'title':[],
        'description':[]
    }

    for item in youtube_response['items']:
      try:
        video_info['videoId'].append(item['id']['videoId'])
        video_info['title'].append(item['snippet']['title'])
        video_info['description'].append(item['snippet']['description'])
      except:
            pass

    video_df = pd.DataFrame(data=video_info)
    # video_df.style.set_properties(**{'text-align': 'left'})
    # print(video_df)
    # result which will be used as input to llm.
    youtube_result = ' '.join(video_df.apply(lambda row: "Video Title: \"" + row['title'] +  "\". Video description: \"" + row['description'] + "\".", axis=1))


    # Call LM to summarize 
    # instruction = "You are a crypto expert"
    query = "Help me summarize these content: " + youtube_result

    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(query)

    # print(response.text)
    st.write("Youtube search result: "+youtube_result[:100] + "...")
    st.write(response.text)
