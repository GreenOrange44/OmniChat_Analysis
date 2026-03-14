import streamlit as st
import pandas as pd
import re
import json
from groq import Groq

import pandas as pd

def sample_chat_for_llm(df, selected_user, max_chars=12000):
    """Safely extracts a contextually rich sample using dynamic fair-share sampling."""
    df_clean = df.dropna(subset=['message']).copy()
    df_clean['message'] = df_clean['message'].astype(str)
    
    # Filter out junk and standard omitted tags
    junk_patterns = r'<Media omitted>|This message was deleted|image omitted|video omitted|document omitted|sticker omitted|audio omitted|GIF omitted|http://|https://'
    df_clean = df_clean[~df_clean['message'].str.contains(junk_patterns, case=False, na=False)]
    df_clean = df_clean[df_clean['message'].str.split().str.len() > 2] # Drop 1-2 word texts
    
    if df_clean.empty:
        return ""

    if selected_user == 'Overall':
        # --- DYNAMIC INCLUSIVE GROUP SAMPLING ---
        # 1. Include ALL users who have sent at least 3 meaningful messages
        user_counts = df_clean['users'].value_counts()
        active_users = user_counts[user_counts >= 3].index
        
        if len(active_users) == 0:
            return ""
        
        # 2. Calculate a "Fair Share" message quota per user
        # We aim for ~250 messages total, divided equally among all active users
        target_total_messages = 250
        msgs_per_user = max(2, target_total_messages // len(active_users))
        longest_quota = msgs_per_user // 2
        random_quota = msgs_per_user - longest_quota
        
        sampled_blocks = []
        for user in active_users:
            user_df = df_clean[df_clean['users'] == user]
            
            # Apply dynamic quotas safely
            user_longest = user_df.assign(length=user_df['message'].str.len()).nlargest(longest_quota, 'length')
            user_remaining = user_df.drop(user_longest.index)
            
            actual_random_quota = min(random_quota, len(user_remaining))
            user_random = user_remaining.sample(actual_random_quota)
            
            sampled_blocks.append(pd.concat([user_longest, user_random]))
            
        # 3. Combine and shuffle thoroughly to simulate a natural flowing chat
        sampled_df = pd.concat(sampled_blocks).sample(frac=1).reset_index(drop=True)
        
    else:
        # --- ISOLATED PERSONAL SAMPLING + MENTIONS BY OTHERS ---
        
        # 1. Grab the selected user's own texts
        user_df = df_clean[df_clean['users'] == selected_user]
        
        longest_msgs = pd.DataFrame()
        random_msgs = pd.DataFrame()
        
        if not user_df.empty:
            longest_msgs = user_df.assign(length=user_df['message'].str.len()).nlargest(100, 'length')
            remaining_pool = user_df.drop(longest_msgs.index)
            random_msgs = remaining_pool.sample(min(100, len(remaining_pool)))
            
        # 2. Grab what OTHER people are saying about them (The Gossip)
        # We extract just their first name (e.g., if the contact is "Swayam Chaurasia", we search for "Swayam")
        first_name = selected_user.split()[0].lower()
        
        # Filter for messages NOT sent by the selected user
        others_df = df_clean[df_clean['users'] != selected_user]
        
        # Find messages where someone else typed their first name
        mentions_df = others_df[others_df['message'].str.lower().str.contains(first_name, na=False, regex=False)]
        
        # Take up to 30 random times they were mentioned by others
        sampled_mentions = pd.DataFrame()
        if not mentions_df.empty:
            sampled_mentions = mentions_df.sample(min(30, len(mentions_df)))
            
        # 3. Combine their texts with the gossip, and shuffle it
        sampled_df = pd.concat([longest_msgs, random_msgs, sampled_mentions]).sample(frac=1).reset_index(drop=True)
        
        if sampled_df.empty:
            return ""

    # --- THE SMART TOKEN GUARDRAIL ---
    # Instead of brutally slicing the string, we build it message by message.
    chat_text = ""
    for _, row in sampled_df.iterrows():
        msg_string = f"{row['users']}: {row['message']}\n"
        
        # If adding this next message exceeds our limit, stop completely!
        if len(chat_text) + len(msg_string) > max_chars:
            chat_text += "...[Chat truncated elegantly for API limits]"
            break
            
        chat_text += msg_string
        
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
    Also can add a brief description for each topic that captures the essence of what they say about it (e.g., "They complain about how bad the food is in the canteen" or "They are obsessed with planning their weekend outings"). Also can add specific back-and-forth examples from the chat to support your analysis.
    
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
def get_user_persona(chat_text, user_name, user_stats, group_stats, api_key):
    """Generates a deep profile using individual stats benchmarked against group stats."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough texts to analyze."}
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert behavioral psychologist and conversational analyst.
    The chat data provided contains messages sent by {user_name}, AND messages from OTHER people in the group who are mentioning or replying to them.
    Analyze {user_name}'s texting style using BOTH their chat history and their hard statistical metrics.
    Consider yet don't over rely on their most used words, average response time, message length, emoji usage, and any other relevant stats to create a nuanced profile of their personality as it manifests in this group chat.
    You MUST extract 1 to 5 direct, verbatim quotes from the provided chat data that perfectly illustrate their archetype and communication style.
    
    INDIVIDUAL METRICS FOR {user_name}:
    {user_stats}
    
    OVERALL GROUP BASELINE (Use this to compare and judge their behavior):
    {group_stats}
    
    Provide a deep, multi-faceted psychological profile. Compare their behavior to the group norm (e.g., do they talk more than average? Are they faster/slower to reply?).
    You MUST respond in valid JSON format exactly like this:
    {{
      "archetype": "A 2-3 word title (e.g., The Anxious Planner, The Ghost, The Hype-Man)",
      "core_traits": ["Trait 1", "Trait 2", "Trait 3"],
      "signature_quotes": ["Verbatim Quote 1", "Verbatim Quote 2", ...],
      "social_role": "4-8 sentences explaining their role compared to the group baseline (can give examples on how others interact with them).",
      "communication_style": "4-8 sentences detailing how they structure their texts. You can give specific examples from their texts to support your analysis.",
      "top_interests": ["Topic 1", "Topic 2", ...]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Chat data:\n{chat_text}"}],
            model="llama-3.1-8b-instant", response_format={"type": "json_object"}, temperature=0.4
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_output": getattr(e, 'response', 'No raw output')}

 
@st.cache_data(show_spinner=False)
def get_user_roast(chat_text, user_name, user_stats, group_stats, api_key):
    """Generates a comedic roast grounded in comparative statistical data."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough texts to roast."}
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a hilarious, sarcastic, and brutally honest comedian. The chat data provided contains messages sent by {user_name}, AND messages from OTHER people in the group who are mentioning or replying to them.
    
    You can roast them about their name, their messages, their personality, their role in the group, their texting habits, how others interact with them, or anything else that stands out about them.

    You can use these hard statistical metrics as ammunition for your roast though don't over-rely on them.:
    {user_stats}
    You can also reference the overall group baseline stats to make comparisons (e.g., "You text 3 times more than the average person in this group, are you unemployed?").
    OVERALL GROUP BASELINE:
    {group_stats}

    Don't try to be nice or politically correct, be brutal in you roast.
    The roast should feel natural and not like a dry recitation of stats.
    
    You can add specific examples from their texts to make the roast more personal and hilarious.
    Look at what OTHER people in the chat are saying about {user_name}. If the group is mocking them, getting annoyed by them, or calling them out, use that gossip to fuel your roast!
    
    You MUST respond in valid JSON format exactly like this:
    {{
      "brutal_roast": "A 5-8 sentence hilarious brutal roast comparing them to the group. Use specific stats and examples (on how they interact or others interact with them) to make it personal and grounded in their actual texting behavior.",
      "biggest_red_flag": "Their worst texting habit. This should be something that genuinely stands out as annoying or cringey in their texts",
      "receipts": [
        {{
          "quote": "The exact verbatim text message here, with no added punctuation",
          "sender": "Name of the person who sent it (Can be from them, or from someone else mocking them)"
        }},
        {{
          "quote": "Another exact verbatim text message (Can be from them, or from someone else mocking them)",
          "sender": "Exact Name of the sender"
        }},
        {{
          "quote": "Verbatim text possibly from a DIFFERENT person in the group who is mocking or talking about {user_name}",
          "sender": "Exact Name of the other person"
        }},
        ],
      "biggest_green_flag": "One backhandedly and sarcastically nice thing about their texts. This should be something that seems like a compliment but is actually a subtle jab"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Messages from {user_name}:\n{chat_text}"}],
            model="llama-3.1-8b-instant", response_format={"type": "json_object"}, temperature=0.7 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "raw_output": getattr(e, 'response', 'No raw output')}



@st.cache_data(show_spinner=False)
def get_group_superlatives(chat_text, users_list, api_key):
    """Assigns funny yearbook-style superlatives to specific group members."""
    if not chat_text or chat_text.strip() == "":
        return {"error": "Not enough text data to assign superlatives."}
        
    client = Groq(api_key=api_key)
    
    prompt = """
You are analyzing a group chat (WhatsApp/Telegram/Discord style) and creating a funny "Group Chat Awards" summary.

Read the provided messages and identify the personalities of the most prominent users in the chat. The prominent users in this chat are: {users_list}

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
      "defining_quote": "A direct quote from the user demonstrating this."
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