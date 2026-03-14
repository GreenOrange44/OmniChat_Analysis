import streamlit as st
import pandas as pd
import re
import json
from groq import Groq

def sample_chat_for_llm(df, selected_user):
    """Safely extracts a contextually rich sample of the chat using stratified or isolated sampling."""
    df_clean = df.dropna(subset=['message']).copy()
    df_clean['message'] = df_clean['message'].astype(str)
    
    # Filter out junk
    junk_patterns = r'<Media omitted>|This message was deleted|image omitted|video omitted|document omitted|sticker omitted|audio omitted|GIF omitted|http://|https://'
    df_clean = df_clean[~df_clean['message'].str.contains(junk_patterns, case=False, na=False)]
    df_clean = df_clean[df_clean['message'].str.split().str.len() > 2] # Drop 1-2 word texts like "ok"
    
    if df_clean.empty:
        return ""

    if selected_user == 'Overall':
        # --- STRATIFIED GROUP SAMPLING ---
        # 1. Find the top 10 most active users (so we don't sample someone who just said "hi" once)
        top_users = df_clean['users'].value_counts().nlargest(10).index
        
        sampled_blocks = []
        for user in top_users:
            user_df = df_clean[df_clean['users'] == user]
            
            # Take up to 15 longest and 15 random messages PER USER (Ensures total representation)
            user_longest = user_df.assign(length=user_df['message'].str.len()).nlargest(15, 'length')
            user_remaining = user_df.drop(user_longest.index)
            user_random = user_remaining.sample(min(15, len(user_remaining)))
            
            sampled_blocks.append(pd.concat([user_longest, user_random]))
            
        # Combine all user samples and shuffle them so it looks like a dynamic chat
        sampled_df = pd.concat(sampled_blocks).sample(frac=1)
        
    else:
        # --- ISOLATED PERSONAL SAMPLING ---
        # Only look at this specific user's texts for their Roast/Profile
        user_df = df_clean[df_clean['users'] == selected_user]
        
        if user_df.empty:
            return ""
            
        # Grab up to 100 of their longest and 100 random texts
        longest_msgs = user_df.assign(length=user_df['message'].str.len()).nlargest(100, 'length')
        remaining_pool = user_df.drop(longest_msgs.index)
        random_msgs = remaining_pool.sample(min(100, len(remaining_pool)))
        
        sampled_df = pd.concat([longest_msgs, random_msgs])

    # Combine into the final text block
    chat_text = "\n".join(sampled_df['users'] + ": " + sampled_df['message'])
    
    # THE TOKEN GUARDRAIL: Hard cut-off at 15,000 characters to prevent Groq API crashes
    if len(chat_text) > 15000:
        chat_text = chat_text[:15000] + "\n...[Chat truncated for API limits]"
        
    return chat_text

@st.cache_data(show_spinner=False)
def get_group_topics(chat_text, group_stats, api_key):
    """Calls Groq API and forces JSON format, grounded in hard lexical stats."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough valid text messages found to analyze."}
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert conversational analyst. Read the following sample of a group chat and the overall statistical metrics. 
    The text may contain Hinglish, slang, and internet shorthand.
    
    CRITICAL STATISTICAL CONTEXT:
    {group_stats}
    
    Your task is to identify the top 4 distinct topics discussed. 
    WARNING: You must base your topics on the ACTUAL VOLUME of conversation. Do not over-index on niche technical jargon, academic terms, or rare professional words just because they stand out to you. Look closely at the "Top 10 Most Used Words" provided in the stats to ground your understanding of what this group actually spends their time talking about (e.g., daily college life, food, gossip, making plans).
    
    You MUST respond in valid JSON format exactly like this:
    {{
      "summary": "A 2-sentence overall summary of the group's dynamic.",
      "topics": [
        {{"name": "Topic Name (e.g., College Drama, Weekend Plans, Exam Stress)", "percentage": 40, "description": "Brief context on what they said about this"}}
      ]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Chat data:\n{chat_text}"}
            ],
            model="llama-3.1-8b-instant", 
            response_format={"type": "json_object"},
            temperature=0.3 # Keep this low so it respects the statistics rather than hallucinating
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_output": getattr(e, 'response', 'No raw output')}

@st.cache_data(show_spinner=False)
def get_user_persona(chat_text, user_name, user_stats, api_key):
    """Generates a deep psychological profile using text AND statistical data."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough texts to analyze."}
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert behavioral psychologist and conversational analyst.
    Analyze {user_name}'s texting style using BOTH their chat history and their hard statistical metrics.
    
    Statistical Metrics for {user_name}:
    {user_stats}
    
    Provide a deep, multi-faceted psychological profile. 
    You MUST respond in valid JSON format exactly like this:
    {{
      "archetype": "A 2-3 word title (e.g., The Anxious Planner, The Ghost, The Hype-Man)",
      "core_traits": ["Trait 1", "Trait 2", "Trait 3"],
      "social_role": "1-2 sentences explaining their role in the group dynamic based on their initiation/reply rates.",
      "communication_style": "1-2 sentences detailing how they structure their texts (length, punctuation, tone).",
      "top_interests": ["Topic 1", "Topic 2", "Topic 3"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Chat data:\n{chat_text}"}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.4 # Keep it slightly lower for analytical accuracy
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_output": getattr(e, 'response', 'No raw output')}
    

@st.cache_data(show_spinner=False)
def get_user_roast(chat_text, user_name, user_stats, api_key):
    """Generates a comedic roast grounded in hard statistical data."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough texts to roast. They are too quiet!"}
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a hilarious, sarcastic, and brutally honest comedian. Read the following chat messages sent by {user_name}.
    
    You MUST use these hard statistical metrics as ammunition for your roast:
    {user_stats}
    
    Roast their texting habits, their response times (call them a ghost if it's high), their "Vibe Kill Rate" (how often they end conversations), and their overall tone. Make it hyper-specific to their stats and text examples.
    
    You MUST respond in valid JSON format exactly like this:
    {{
      "brutal_roast": "A 3-4 sentence hilarious and brutal roast incorporating their stats.",
      "biggest_red_flag": "Their worst texting habit (reference a stat if possible).",
      "biggest_green_flag": "One genuinely nice (or backhandedly nice) thing about their texts."
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
            temperature=0.7 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_output": getattr(e, 'response', 'No raw output')}

@st.cache_data(show_spinner=False)
def get_group_superlatives(chat_text, api_key):
    """Assigns funny yearbook-style superlatives to group members."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough text data to assign superlatives."}
        
    client = Groq(api_key=api_key)
    
    prompt = """
You are analyzing a group chat (WhatsApp/Telegram/Discord style) and creating a funny "Group Chat Awards" summary.

Read the provided messages and identify the personalities of the most prominent users in the chat.

Assign up to 5 humorous titles that reflect common group chat behavior. The humor should feel natural to Indian chat culture.

Examples of titles (do NOT limit yourself to these):
- "The Good Morning Broadcaster"
- "Last-Minute Assignment Guy"
- "The Silent Reader (Seen-Zoned Everyone)"
- "The Plan Canceller"
- "The Night Owl"
- "The Over-Analyzer"
- "The Sticker Machine"
- "The Random 2AM Philosopher"
- "The Yap Machine"

Rules:
- Choose only the most active or distinctive personalities.
- The title should be short and funny.
- The reason should be a single sentence referencing their typical behavior in the messages.
- Do NOT invent events not present in the messages.
- If fewer than 5 personalities are clearly visible, return fewer.

You MUST respond in valid JSON format exactly like this:

{
  "superlatives": [
    {
      "user": "Name of user",
      "title": "Funny Award Title",
      "reason": "One sentence explaining why they got this title based on their chat behavior."
    }
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
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_output": getattr(e, 'response', 'No raw output')}