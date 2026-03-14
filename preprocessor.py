import re
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


def preprocess(data: str, filename: str = "") -> pd.DataFrame:
    # DYNAMIC GROUP NAME
    group_name = ""
    if filename:
        group_name = filename.replace("WhatsApp Chat with ", "").replace(".txt", "").strip()

    # AUTO-DETECT OS FORMAT
    # iOS: [DD/MM/YY, HH:MM:SS]
    ios_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}:\d{2}(?:\s+[AP]M)?\]'
    # Android: DD/MM/YY, HH:MM pm - 
    android_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}(?:\s+[aApP][mM])?\s+-\s+'

    if re.search(ios_pattern, data):
        pattern = ios_pattern
        is_android = False
    elif re.search(android_pattern, data):
        pattern = android_pattern
        is_android = True
    else:
        # Stop the app gracefully with a UI error instead of a Python crash
        st.error("🚨 Unsupported file format. Please upload a standard WhatsApp exported .txt file.")
        st.stop()

    # SPLIT DATA
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # CLEAN DATES & PARSE DATETIME
    if is_android:
        # Strip the trailing hyphen and spaces from Android dates
        df['message_date'] = df['message_date'].str.replace(r'\s+-\s*$', '', regex=True)
    else:
        # Strip the brackets from iOS dates
        df['message_date'] = df['message_date'].str.strip('[]')

    # Pandas 'mixed' format solves all 12hr/24hr and DD/MM vs MM/DD headaches!
    df['message_date'] = pd.to_datetime(df['message_date'], format='mixed', dayfirst=True, errors='coerce')

    # EXTRACT USERS & MESSAGES
    extracted = df['user_message'].str.extract(r'^([^:]+):\s(.*)')
    df['users'] = extracted[0]
    df['message'] = extracted[1]

    # DROP INVALID ROWS (This instantly removes system alerts)
    df = df.dropna(subset=['message_date', 'users', 'message'])

    # FILTER AI BOTS & ANNOUNCEMENTS
    df = df[df['users'].str.strip() != 'You']
    df = df[df['users'].str.strip() != 'Meta AI']
    if group_name:
        df = df[df['users'].str.strip() != group_name]
    
    # Clean up any trailing whitespace or newlines in the messages
    df['message'] = df['message'].str.strip()

    df.drop(columns=['user_message'], inplace=True)

    df['full_date'] = df['message_date'].dt.date
    df['year'] = df['message_date'].dt.year
    df['month_num'] = df['message_date'].dt.month
    df['month'] = df['message_date'].dt.month_name()
    df['day'] = df['message_date'].dt.day
    df['day_name'] = df['message_date'].dt.day_name()
    df['hour'] = df['message_date'].dt.hour
    df['minute'] = df['message_date'].dt.minute
    df['second'] = df['message_date'].dt.second

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(f"00-01")
        else:
            period.append(f"{hour:02d}-{(hour+1):02d}")

    df['period'] = period

    sia = SentimentIntensityAnalyzer()

    def calculate_sentiment(msg):
        # Ignore media and deleted messages
        if msg.strip() in ['<Media omitted>', 'This message was deleted.', 'null']:
            return 0.0, 0.0, 0.0, 0.0
            
        scores = sia.polarity_scores(msg)
        return scores['pos'], scores['neg'], scores['neu'], scores['compound']

    # Apply the function and expand the results into new columns
    # VADER's compound score ranges from -1 (extremely negative) to +1 (extremely positive)
    sentiment_scores = df['message'].apply(calculate_sentiment)
    
    df['pos_score'] = [score[0] for score in sentiment_scores]
    df['neg_score'] = [score[1] for score in sentiment_scores]
    df['neu_score'] = [score[2] for score in sentiment_scores]
    df['compound_score'] = [score[3] for score in sentiment_scores]

    # Categorize the compound score for easier filtering
    def categorize_mood(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'
            
    df['mood'] = df['compound_score'].apply(categorize_mood)


    return df
