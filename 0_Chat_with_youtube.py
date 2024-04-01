import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

import google.generativeai as genai
import googleapiclient.discovery
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json


with st.sidebar:
    google_api_key = st.text_input(
        "Google API Key", key="google_api_key", type="password"
    )
    "Get Google API key for Youtube and Gemini. You can get one here for free: https://console.cloud.google.com"
with st.sidebar:
    youtube_max_results = st.number_input(
        "Max results in Youtube", 1
    )
    "Top N results in Youtube search. suggest < 50"
with st.sidebar:
    final_summary_text_length = st.slider('The words limit for the final summary', 100, 1000, 300)

# for openAi

# with st.sidebar:
#     openai_api_key = st.text_input(
#         "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
#     )
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 Chat with Youtube")

# """
# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
# """

"""
In this example, we will summarize the videos from Youtube search results. The summary is based on title, description, captions if available. We will show summary for each video first. Then show a comprehensive summary for all videos. 
"""


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the Youtube. Input the query for Youtube, I will help you to summarize the content."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="E.g. Top stocks"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check key
    if not google_api_key:
        st.info("Please add your Google API key to continue. You can get one here for free: https://console.cloud.google.com")
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
    # st.write(json_string)
    
    # save in dataframe, so we can look at it as the source
    video_info = {
        'videoId':[],
        'title':[],
        'description':[],
        'captions':[]
    }

    for item in youtube_response['items']:
      try:
        video_info['videoId'].append(item['id']['videoId'])
        # st.write("videoId: " + item['id']['videoId'] )
        video_info['title'].append(item['snippet']['title'] or "empty")
        # st.write("Title: " + item['snippet']['title'])
        video_info['description'].append(item['snippet']['description'] or "empty")
        # get En caption
        captions_concat = ""
        try:
            captions_str = YouTubeTranscriptApi.get_transcript(item['id']['videoId'] ,languages=['en'])
            captions_concat = ','.join([item['text'] for item in captions_str])
        except:
            pass
        # st.write(captions_concat)
        video_info['captions'].append(captions_concat)    
      except:
            pass

    # st.write(video_info)
    video_df = pd.DataFrame(data=video_info)
    st.write("Raw data:")
    st.write(video_df)

    st.write("Call LLM....wait a second...")
    # Prepare a string content which will be used as input to llm.
    youtube_result = ' '.join(video_df.apply(lambda row: "Video Title: \"" + row['title'] +  "\". Video description: \"" + row['description'] + "\". Video captions: \"" + row['captions'] + "\".", axis=1))

    
    # Call LM to summarize 
    # instruction = "You are a crypto expert"
    query_each = "Help me summarize these content: " + youtube_result
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-pro')
    response_each = model.generate_content(query_each)
    st.title("Summary of each video: ")
    st.write(response_each.text)

    # prompt
    summary_text_length_str = str(final_summary_text_length)
    query_all = "Given a list of videos on the, provide a comprehensive summary of the key themes, ideas, and insights covered across all the videos as a whole. Please consolidate the information into one cohesive paragraph which is less than " + summary_text_length_str + " words, focusing on the overarching concepts rather than detailing each video separately. The content: " + youtube_result
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-pro')
    response_all = model.generate_content(query_all)
    st.title("Summary of all of these videos in less than " + summary_text_length_str + " words: ")
    st.write(response_all.text)
