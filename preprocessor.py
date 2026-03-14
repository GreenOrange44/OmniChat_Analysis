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

    ios_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}:\d{2}(?:\s+[aApP][mM])?\]'
    android_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}(?:\s+[aApP][mM])?\s+-\s+'

    if re.search(ios_pattern, data):
        pattern = ios_pattern
        is_android = False
    elif re.search(android_pattern, data):
        pattern = android_pattern
        is_android = True
    else:
        st.error("🚨 Unsupported file format. Please upload a standard WhatsApp exported .txt file.")
        st.stop()

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # CLEAN RAW DATE STRINGS
    if is_android:
        df['message_date'] = df['message_date'].str.replace(r'\s+-\s*$', '', regex=True)
    else:
        df['message_date'] = df['message_date'].str.strip('[]')

    df['message_date'] = df['message_date'].str.upper().str.strip()

    #  DYNAMIC DATE FORMAT INFERENCE ---
    # Split the string to isolate just the date part (e.g., "26/08/23")
    date_only = df['message_date'].str.split(',').str[0]
    date_parts = date_only.str.split('/', expand=True)
    
    # Convert the first and second numbers to integers
    p0 = pd.to_numeric(date_parts[0], errors='coerce')
    p1 = pd.to_numeric(date_parts[1], errors='coerce')

    # MATHEMATICAL CHECK:
    # If any 2nd number is greater than 12 (e.g. 3/14/26), it MUST be MM/DD/YY
    if p1.max() > 12:
        date_fmts = ['%m/%d/%y', '%m/%d/%Y']
    # If any 1st number is greater than 12 (e.g. 26/08/23), it MUST be DD/MM/YY
    elif p0.max() > 12:
        date_fmts = ['%d/%m/%y', '%d/%m/%Y']
    # If ambiguous (e.g., all dates are like 05/06/23), default to Rest-of-World standard
    else:
        date_fmts = ['%d/%m/%y', '%d/%m/%Y']

    time_fmts = [', %I:%M %p', ', %H:%M', ', %I:%M:%S %p', ', %H:%M:%S']
    
    # Combine the detected date format with all possible time formats
    formats = [d + t for d in date_fmts for t in time_fmts]

    # --- 4. CASCADING PARSER ---
    parsed_dates = pd.Series(pd.NaT, index=df.index)
    for fmt in formats:
        missing = parsed_dates.isna()
        if not missing.any(): 
            break
        temp = pd.to_datetime(df.loc[missing, 'message_date'], format=fmt, errors='coerce')
        parsed_dates = parsed_dates.fillna(temp)

    df['message_date'] = parsed_dates
    
    # Drop any severely corrupted lines
    df = df.dropna(subset=['message_date'])

    # --- 5. EXTRACT USERS & MESSAGES ---
    extracted = df['user_message'].str.extract(r'^([^:]+):\s(.*)')
    df['users'] = extracted[0]
    df['message'] = extracted[1]

    df = df.dropna(subset=['users', 'message'])

    # Filter Bots
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
