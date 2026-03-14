import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


def preprocess(data: str, filename: str = "") -> pd.DataFrame:
    group_name = ""
    if filename:
        # WhatsApp standard export names are "WhatsApp Chat with [Name].txt"
        # We strip the prefix and suffix to get the exact group name dynamically
        group_name = filename.replace("WhatsApp Chat with ", "").replace(".txt", "").strip()

    #pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M\]'
    pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}:\d{2}(?:\s+[AP]M)?\]'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    df['message_date'] = df['message_date'].str.strip('[]')

    date_12h = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M:%S %p', errors='coerce')
    date_24h = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M:%S', errors='coerce')
    
    # Combine them (if 12h fails, it uses 24h)
    df['message_date'] = date_12h.fillna(date_24h)

    # (Optional fallback just in case the year is 4 digits like 2024 instead of 24)
    if df['message_date'].isnull().any():
        df['message_date'] = pd.to_datetime(df['message_date'], dayfirst=True, errors='coerce')

    # Extract the sender and the message
    extracted = df['user_message'].str.extract(r'^([^:]+):\s(.*)')

    
    df['users'] = extracted[0]
    df['message'] = extracted[1]

    # Clean up junk data (Meta AI and system notifications)
    df = df.dropna(subset=['users', 'message']) # Drop the NaNs immediately.
    df = df[df['users'].str.strip() != 'Meta AI']
    df = df[df['users'].str.strip() != 'You']

    # Dynamically remove messages sent BY the group itself (Announcement groups)
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
