# 📊 OmniChat Analytics: Behavioral insights powered by LLaMA-3.
**Live Demo:** [https://omni-chat-analysis.streamlit.app/]

An advanced, end-to-end conversational analytics engine that transforms raw WhatsApp export data into deep psychological, behavioral, and network insights. Moving beyond standard word-frequency counters, this application leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to map social dynamics, detect sentiment trends, and generate AI-driven communication personas.

## 🗂️ Dashboard Modules & Features

The application is structured into six progressive analytical tabs, moving from broad macro-trends down to deep behavioral and generative AI insights:

### ⏱️ 1. Timelines
Visualizes the macro-lifespan of the chat to identify periods of high and low engagement.
* **Monthly Trend:** A line graph tracking the total volume of messages month-over-month.
* **Daily Trend:** A granular chronological timeline highlighting specific days with unusual spikes in conversation.

### 📅 2. Activity Maps
Maps out the temporal habits of the group to see exactly *when* people are talking.
* **Busiest Days & Months:** Bar charts identifying the peak seasons and days of the week for the group.
* **Weekly Activity Heatmap:** An interactive matrix showing the exact hour blocks (e.g., Friday at 10 PM) when the chat is historically the most active.

### 🔠 3. Lexical Analysis
Cleans and processes the raw text to uncover the group's vocabulary and visual communication style.
* **Dynamic Word Cloud:** Generates a visual representation of the most frequently used terms, utilizing a custom stopword pipeline to filter out meaningless internet slang and standard grammar.
* **Emoji Distribution:** Extracts and tallies emoji usage, displaying the top choices in a clean data table and pie chart.

### 👥 4. User Dynamics (Network & Behavioral Analytics)
The core behavioral engine of the application. It utilizes pandas time-shifting and threshold logic to analyze social interactions.
* **The Interaction Matrix:** A Plotly-powered heatmap detailing the exact "Who replies to Whom" ecosystem to identify sub-groups or cliques.
* **Normalized Initiation Metrics:** Calculates the true "Conversation Starters" and "Vibe Killers" by dividing their initiation count by the total number of conversations they were actually present for, preventing volume bias.
* **Response Telemetry:** Measures average intra-session response times to calculate communication speed and "ghosting" habits.

### 🎭 5. Sentiment Board
Applies VADER (Valence Aware Dictionary and sEntiment Reasoner) to score the emotional polarity of the chat.
* **Overall Mood:** A donut chart breaking down the historical percentage of Positive, Negative, and Neutral messages.
* **Sentiment Timeline & Hourly Trends:** Tracks how the group's mood fluctuates over the months and changes throughout the hours of the day.
* **User Rankings:** Ranks participants from most naturally positive to most negative based on their average compound sentiment score.

### ✨ 6. AI Insights (Powered by Groq & LLaMA-3)
Leverages a robust LLM pipeline with strict JSON formatting to generate human-readable summaries from unstructured, multilingual (e.g., Hinglish), and slang-heavy text.
* **Zero-Shot Topic Modeling:** Dynamically categorizes conversation themes (e.g., Academics, Sports) with accurate percentage breakdowns and context descriptions.
* **User Persona Generation:** Analyzes a specific user's text history to generate a psychological and communication profile, including their "Vibe," texting style, and frequent conversational habits.

## 🛠️ Tech Stack
* **Frontend:** Streamlit, Custom CSS UI Architecture
* **Data Processing:** Pandas, Regex Time-Series Modeling
* **Machine Learning/NLP:** NLTK (VADER), Groq API (LLaMA-3 8B)
* **Data Visualization:** Plotly Interactive Graphs, Seaborn, Matplotlib

## 🚀 Running the Project Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GreenOrange44/OmniChat_Analysis
   cd OmniChat_Analysis

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Set your API key:**
   To use the Generative AI features, obtain a free API key from Groq Console. You can input this key directly in the app's sidebar UI.

4. **Run streamlit server:**
   ```bash
   streamlit run app.py