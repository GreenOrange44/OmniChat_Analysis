import streamlit as st
import pandas as pd
import re
import json
from groq import Groq

def sample_chat_for_llm(df, selected_user, sample_size=150):
    """Safely extracts a contextually rich sample of the chat."""
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
        
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.dropna(subset=['message']).copy()
    
    # Cast to string to prevent NaN/float errors during regex
    df_clean['message'] = df_clean['message'].astype(str)
    
    # Filter out junk (added na=False to prevent ValueError)
    df_clean = df_clean[~df_clean['message'].str.contains('<Media omitted>|This message was deleted', case=False, na=False)]
    df_clean = df_clean[df_clean['message'].str.split().str.len() > 2]
    patterns = ['image omitted', 'sticker omitted',  'audio omitted', 'video omitted', 'GIF omitted', 'http://', 'https://']

    df_clean= df_clean[df_clean['message'].apply(lambda x: not any(pattern in x for pattern in patterns))]
    
    # If the filter wiped out everything, return empty
    if df_clean.empty:
        return ""

    # Grab the most contextual messages + a random spread
    longest_msgs = df_clean.assign(length=df_clean['message'].str.len()).nlargest(150, 'length')
    
    # Remove those exact messages from the main dataframe pool using their index
    remaining_pool = df_clean.drop(longest_msgs.index)
    
    # Take a random spread from whatever is left
    random_msgs = remaining_pool.sample(min(150, len(remaining_pool)))
    
    # Combine them (no need for drop_duplicates anymore!)
    sampled_df = pd.concat([longest_msgs, random_msgs])
    
    sampled_df = pd.concat([longest_msgs, random_msgs]).drop_duplicates()
    chat_text = "\n".join(sampled_df['users'] + ": " + sampled_df['message'])
    
    return chat_text

@st.cache_data(show_spinner=False)
def get_group_topics(chat_text, api_key):
    """Calls Groq API and forces JSON format."""
    # Catch empty text BEFORE the API call
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough valid text messages found to analyze. They might be too short or mostly media."}
        
    client = Groq(api_key=api_key)
    
    prompt = """
    You are an expert conversational analyst. Read the following sample of a group chat. The text may contain Hinglish, slang, and internet shorthand. 
    Analyze the text and return the top 4 distinct topics discussed. 
    You MUST respond in valid JSON format exactly like this:
    {
      "summary": "A 2-sentence overall summary of the group's dynamic.",
      "topics": [
        {"name": "Topic Name (e.g., Academics, Tech, Sports)", "percentage": 40, "description": "Brief context on what they said about this"}
      ]
    }
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Chat data:\n{chat_text}"}
            ],
            model="llama-3.1-8b-instant", 
            response_format={"type": "json_object"},
            temperature=0.3
        )
        raw_content = response.choices[0].message.content
        print("\n--- RAW API RESPONSE ---")
        print(raw_content)
        print("------------------------\n")
        
        # 1. Strip out markdown code blocks if the LLM added them
        cleaned_content = raw_content.replace('```json', '').replace('```', '').strip()
        
        # 2. Parse the clean JSON
        parsed_json = json.loads(cleaned_content)
        return parsed_json
        
    except json.JSONDecodeError as e:
        # If it STILL fails to parse, return the raw text so we can see it in the app!
        print(f"JSON Parse Error: {e}")
        return {"error": "Failed to parse JSON", "raw_output": raw_content}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def get_user_persona(chat_text, user_name, api_key):
    """Generates a psychological/communication profile of a specific user."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough valid text messages found to analyze. They might be too short or mostly media."}

        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    Analyze the provided chat messages sent by {user_name}. The text may contain Hinglish and internet slang.
    Provide a psychological and communication profile of this user.
    You MUST respond in valid JSON format exactly like this:
    {{
      "vibe": "3-word description of their energy (e.g., Chill, Anxious Planner, Sarcastic)",
      "communication_style": "1-sentence description of how they text.",
      "frequent_habits": ["habit 1", "habit 2"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Messages from {user_name}:\n{chat_text}"}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.4
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}