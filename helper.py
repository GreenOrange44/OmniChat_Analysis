import pandas as pd
import re
from collections import Counter
import emoji
from urlextract import URLExtract
extractor = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    # 1. Total Messages
    num_messages = df.shape[0]

    # 2. Total Words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # 3. SMART MEDIA COUNTING
    # Create a dictionary to hold whatever we find
    media_dict = {
        'Images': 0,
        'Videos': 0,
        'GIFs': 0,
        'Audio/Voice Notes': 0,
        'Documents': 0,
        'Stickers': 0,
        'Unspecified Media': 0
    }

    for message in df['message']:
        msg_lower = message.lower()
        
        # Check for Android Generic
        if '<media omitted>' in msg_lower:
            media_dict['Unspecified Media'] += 1
            
        # Check for iOS specific or "Export With Media" filenames
        elif 'image omitted' in msg_lower or '.jpg' in msg_lower or '.png' in msg_lower:
            media_dict['Images'] += 1
        elif 'video omitted' in msg_lower or '.mp4' in msg_lower:
            media_dict['Videos'] += 1
        elif 'gif omitted' in msg_lower or '.gif' in msg_lower:
            media_dict['GIFs'] += 1
        elif 'audio omitted' in msg_lower or '.opus' in msg_lower or '.mp3' in msg_lower:
            media_dict['Audio/Voice Notes'] += 1
        elif 'document omitted' in msg_lower or '.pdf' in msg_lower or '.docx' in msg_lower:
            media_dict['Documents'] += 1
        elif 'sticker omitted' in msg_lower or '.sticker' in msg_lower:
            media_dict['Stickers'] += 1

    # Remove categories that have 0 counts so the UI stays clean
    clean_media_dict = {k: v for k, v in media_dict.items() if v > 0}

    # 4. Links Shared
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))

    return {
        'total_messages': num_messages,
        'total_words': len(words),
        'media_messages': clean_media_dict, # Pass the clean dictionary
        'links_shared': len(links)
    }
    

def fetch_frequent_users(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_counts = df['users'].value_counts().reset_index()
    user_counts.columns = ['users', 'message_count']
    percentages = round((df['users'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'users':'name', 'message_count': 'percentage'})
    return user_counts.head(), percentages

def create_wordcloud(selected_user: str, df: pd.DataFrame):
    from wordcloud import WordCloud
    
    # 1. Improved file reading
    with open('stop-hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(word.strip().lower() for word in f.read().split())

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    temp = df[df['users'] != 'group_notification']
    patterns = ['image omitted', 'sticker omitted', 'audio omitted', 'video omitted', 'GIF omitted', 'http://', 'https://', '<Media omitted>']
    temp = temp[temp['message'].apply(lambda x: not any(pattern in x for pattern in patterns))]
    
    # 2. Add collocations=False and pass stopwords directly
    wc = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color='white',
        stopwords=stop_words, # Critical
        collocations=False    # Prevents word pairs from bypassing filters
    )
    
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user: str,df: pd.DataFrame) -> pd.DataFrame:

    f = open('stop-hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    temp = df[df['users'] != 'group_notification']
    patterns = ['image omitted', 'sticker omitted',  'audio omitted', 'video omitted', 'GIF omitted', 'http://', 'https://', '<Media omitted>']

    temp = temp[temp['message'].apply(lambda x: not any(pattern in x for pattern in patterns))]

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user: str,df: pd.DataFrame) -> pd.DataFrame:

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    daily_timeline = df.groupby('full_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Convert day_name to categorical with the desired order
    df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)

    # Create pivot table; the index will now follow the categorical order
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def mood_breakdown(selected_user, df):
    """Calculates the total count of Positive, Negative, and Neutral messages."""
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
        
    mood_counts = df['mood'].value_counts().reset_index()
    mood_counts.columns = ['Mood', 'Count']
    return mood_counts

def monthly_sentiment(selected_user, df):
    """Calculates the average sentiment score for each month."""
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
        
    # Group by year and month_num to maintain chronological order
    sentiment_df = df.groupby(['year', 'month_num', 'month'])['compound_score'].mean().reset_index()
    
    time = []
    for i in range(sentiment_df.shape[0]):
        time.append(sentiment_df['month'][i] + "-" + str(sentiment_df['year'][i]))
    
    sentiment_df['time'] = time
    return sentiment_df

def hourly_sentiment(selected_user, df):
    """Calculates the average sentiment score for each hour of the day."""
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
        
    hourly_scores = df.groupby('hour')['compound_score'].mean().reset_index()
    return hourly_scores

def user_sentiment_ranking(df):
    """Ranks users by their average compound sentiment score (Overall only)."""
    # Exclude system notifications from the ranking
    temp_df = df[df['users'] != 'group_notification']
    
    user_scores = temp_df.groupby('users')['compound_score'].mean().reset_index()
    # Sort from most positive to most negative
    user_scores = user_scores.sort_values(by='compound_score', ascending=True) 
    return user_scores


def behavioral_analysis(df, session_threshold_minutes=120):
    df = df.sort_values('message_date').reset_index(drop=True)
    df['time_diff'] = df['message_date'].diff()
    
    threshold = pd.Timedelta(minutes=session_threshold_minutes)
    df['is_new_session'] = df['time_diff'] > threshold
    df.loc[0, 'is_new_session'] = True 
    df['session_id'] = df['is_new_session'].cumsum()
    
    # 1. Find how many total unique sessions each user participated in
    user_participation = df.groupby('users')['session_id'].nunique().reset_index()
    user_participation.columns = ['User', 'Total_Sessions_Present']
    
    # Filter out "ghost" users who barely speak to avoid skewed 100% ratios (min 5 sessions)
    valid_users = user_participation[user_participation['Total_Sessions_Present'] >= 5]
    
    # --- STARTERS (NORMALIZED) ---
    sessions = df.groupby('session_id')
    raw_starters = sessions.first()['users'].value_counts().reset_index()
    raw_starters.columns = ['User', 'Started_Count']
    
    # Merge and calculate ratio
    starters = pd.merge(valid_users, raw_starters, on='User', how='left').fillna(0)
    starters['Starter Rate (%)'] = ((starters['Started_Count'] / starters['Total_Sessions_Present']) * 100).round(1)
    starters = starters.sort_values(by='Starter Rate (%)', ascending=False)
    
    # --- KILLERS (NORMALIZED) ---
    raw_killers = sessions.last()['users'].value_counts().reset_index()
    raw_killers.columns = ['User', 'Killed_Count']
    
    # Merge and calculate ratio
    killers = pd.merge(valid_users, raw_killers, on='User', how='left').fillna(0)
    killers['Killer Rate (%)'] = ((killers['Killed_Count'] / killers['Total_Sessions_Present']) * 100).round(1)
    killers = killers.sort_values(by='Killer Rate (%)', ascending=False)
    
    # --- REPLY MATRIX & GHOSTING 
    df['prev_user'] = df['users'].shift(1)
    interactions = df[df['users'] != df['prev_user']].dropna(subset=['prev_user'])
    reply_matrix = pd.crosstab(interactions['prev_user'], interactions['users'])
    
    intra_session_replies = interactions[interactions['time_diff'] < threshold].copy()
    intra_session_replies['response_time_mins'] = intra_session_replies['time_diff'].dt.total_seconds() / 60
    
    avg_response_time = intra_session_replies.groupby('users')['response_time_mins'].mean().reset_index()
    avg_response_time.columns = ['User', 'Avg Response Time (Mins)']
    avg_response_time = avg_response_time.sort_values('Avg Response Time (Mins)')

    # Get the total number of sessions that happened in the ENTIRE group
    total_group_sessions = df['session_id'].nunique()

    # Add the "Group Share" which WILL sum to exactly 100%
    starters['Group Share (%)'] = ((starters['Started_Count'] / total_group_sessions) * 100).round(1)
    
    killers['Group Share (%)'] = ((killers['Killed_Count'] / total_group_sessions) * 100).round(1)

    # Return the updated dataframes (make sure to include the new column!)
    return starters[['User', 'Starter Rate (%)', 'Group Share (%)']], killers[['User', 'Killer Rate (%)', 'Group Share (%)']], reply_matrix, avg_response_time