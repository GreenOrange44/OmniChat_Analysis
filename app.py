import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import llm_helper
import plotly.express as px
import plotly.graph_objects as go

plt.style.use('dark_background')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Add a bit of breathing room and round the corners of standard elements */
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    /* Make the metric cards look more like a dashboard */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #25D366; /* WhatsApp Green */
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg", width=60)
    st.title('Chat Analyzer')
    st.markdown("Upload your exported WhatsApp chat `.txt` file to get started.")

    st.markdown("### 🤖 Advanced AI Features")
    api_key = st.text_input("Enter Groq API Key (Optional)", type="password", help="Get a free key at console.groq.com to unlock LLM insights.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])

# --- MAIN APP LOGIC ---
if uploaded_file is not None:
    # Process Data
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)

    # Fetch users
    user_list = df['users'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification') 
    user_list.sort()
    user_list.insert(0, 'Overall')
    
    st.sidebar.markdown("---")
    selected_user = st.sidebar.selectbox('👤 Analyze user:', user_list)

    # Initialize the main app state if it doesn't exist
    if 'main_analysis_triggered' not in st.session_state:
        st.session_state['main_analysis_triggered'] = False

    # When clicked, lock it to True!
    if st.sidebar.button('🚀 Generate Analysis', use_container_width=True):
        st.session_state['main_analysis_triggered'] = True
        # Reset the LLM analysis state when a new analysis is triggered
        st.session_state['show_llm_analysis'] = False

    if st.session_state.get('main_analysis_triggered', False):
        
        # Fetch stats
        stats = helper.fetch_stats(selected_user, df)
        
        # --- HERO SECTION: TOP LEVEL METRICS ---
        st.title(f"📊 Chat Analysis: {selected_user}")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Messages", value=stats['total_messages'])
        with col2:
            st.metric(label="Total Words", value=stats['total_words'])
        with col3:
            # Formatting media messages nicely
            media_str = " | ".join([f"{k.capitalize()}: {v}" for k, v in stats['media_messages'].items()])
            st.metric(label="Media Shared", value=sum(stats['media_messages'].values()), help=media_str)
        with col4:
            st.metric(label="Links Shared", value=stats['links_shared'])

        st.markdown("---")

        # --- TABS FOR PROGRESSIVE DISCLOSURE ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⏱️ Timelines", "📅 Activity Maps", "🔠 Lexical Analysis", "👥 User Dynamics", "🎭 Sentiment", "✨ AI Insights"])
        with tab1:
            st.subheader("Message Volume Over Time")
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.caption("Monthly Trend")
                timeline = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(timeline['time'], timeline['message'], color='#25D366', linewidth=2)
                plt.xticks(rotation=45)
                fig.patch.set_alpha(0.0) # Transparent background
                st.pyplot(fig)
                
            with col_t2:
                st.caption("Daily Trend")
                daily_timeline = helper.daily_timeline(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(daily_timeline['full_date'], daily_timeline['message'], color='#3B82F6', linewidth=1.5)
                plt.xticks(rotation=45)
                fig.patch.set_alpha(0.0)
                st.pyplot(fig)

        with tab2:
            st.subheader("When are users most active?")
            col_a1, col_a2 = st.columns(2)

            with col_a1:
                st.caption("Busiest Day of the Week")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(busy_day.index, busy_day.values, color='#8B5CF6')
                plt.xticks(rotation=45)
                fig.patch.set_alpha(0.0)
                st.pyplot(fig)

            with col_a2:
                st.caption("Busiest Month of the Year")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(busy_month.index, busy_month.values, color='#F59E0B')
                plt.xticks(rotation=45)
                fig.patch.set_alpha(0.0)
                st.pyplot(fig)

            st.caption("Weekly Activity Heatmap (Hour vs Day)")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(user_heatmap, cmap="Greens", ax=ax, annot_kws={"color": "white"})
            fig.patch.set_alpha(0.0)
            st.pyplot(fig)

        with tab3:
            st.subheader("What are people saying?")
            col_w1, col_w2 = st.columns([2, 1]) # Make wordcloud wider than dataframe
            
            with col_w1:
                st.caption("Word Cloud")
                wordcloud = helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                fig.patch.set_alpha(0.0)
                st.pyplot(fig)
                
            with col_w2:
                st.caption("Top 10 Words")
                most_common_df = helper.most_common_words(selected_user, df)
                # Ensure the columns have clean names for the dataframe display
                most_common_df.columns = ['Word', 'Frequency']
                st.dataframe(most_common_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Emoji Usage")
            emoji_df = helper.emoji_helper(selected_user, df)
            
            if not emoji_df.empty:
                col_e1, col_e2 = st.columns([1, 2])
                with col_e1:
                    emoji_df.columns = ['Emoji', 'Count']
                    st.dataframe(emoji_df.head(10), use_container_width=True, hide_index=True)
                with col_e2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # Plot only top 5 for a cleaner pie chart
                    top_emojis = emoji_df.head(5)
                    ax.pie(top_emojis['Count'], labels=top_emojis['Emoji'], autopct="%0.1f%%", colors=sns.color_palette("pastel"))
                    fig.patch.set_alpha(0.0)
                    st.pyplot(fig)
            else:
                st.info("No emojis found for this user.")

        with tab4:
            if selected_user == 'Overall':
                st.subheader('Most Active Users')
                x, percentages = helper.fetch_frequent_users(df)
                
                col_u1, col_u2 = st.columns([2, 1])
                with col_u1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(x['users'], x['message_count'], color='#EC4899')
                    plt.xticks(rotation=45)
                    fig.patch.set_alpha(0.0)
                    st.pyplot(fig)

                with col_u2:
                    percentages.columns = ['User', 'Message Share (%)']
                    st.dataframe(percentages, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("🕸️ Network & Behavioral Analytics")
                
                # Fetch the behavioral data
                starters, killers, reply_matrix, response_times = helper.behavioral_analysis(df)
                
                # --- SECTION 1: WHO REPLIES TO WHOM ---
                st.markdown("#### The Interaction Matrix")
                st.caption("Read rows left-to-right. E.g., How often did the user on the Y-axis get a reply from the user on the X-axis?")
                
                # Plotly Heatmap
                fig_matrix = px.imshow(
                    reply_matrix, 
                    labels=dict(x="Replied By", y="Message Sent By", color="Interactions"),
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                # Make it blend with the dark theme
                fig_matrix.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"))
                st.plotly_chart(fig_matrix, use_container_width=True)
                
                st.markdown("---")
                
                # --- SECTION 2: THE INSTIGATORS AND THE GHOSTS ---
                col_b1, col_b2, col_b3 = st.columns(3)
                
                with col_b1:
                    st.markdown("#### 🗣️ True Initiators")
                    st.caption("% of times they started the chat when present.")
                    st.dataframe(
                        starters.head(10), 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Starter Rate (%)": st.column_config.ProgressColumn(
                                "Initiation Rate",
                                help="Percentage of participated conversations they started",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            )
                        }
                    )
                    
                with col_b2:
                    st.markdown("#### 🛑 True Vibe Killers")
                    st.caption("% of times they sent the last message.")
                    st.dataframe(
                        killers.head(10), 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Killer Rate (%)": st.column_config.ProgressColumn(
                                "Kill Rate",
                                help="Percentage of participated conversations they ended",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            )
                        }
                    )
                    
                with col_b3:
                    st.markdown("#### ⏱️ Fastest Responders")
                    st.caption("Average time to text back (minutes).")
                    response_times['Avg Response Time (Mins)'] = response_times['Avg Response Time (Mins)'].round(1)
                    st.dataframe(
                        response_times.head(10), 
                        use_container_width=True, 
                        hide_index=True
                    )
            else:
                st.info("Select 'Overall' in the sidebar to view group dynamics.")

        with tab5:
            st.subheader("Chat Mood & Sentiment Analysis")
            st.markdown("---")

            col_s1, col_s2 = st.columns([1, 2])
            
            # 1. Overall Mood Breakdown (Donut Chart)
            with col_s1:
                st.caption("Overall Message Mood")
                mood_df = helper.mood_breakdown(selected_user, df)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                # Custom colors: Green for Positive, Red for Negative, Gray for Neutral
                colors = {'Positive': '#25D366', 'Negative': '#EF4444', 'Neutral': '#64748B'}
                pie_colors = [colors.get(mood, '#000000') for mood in mood_df['Mood']]
                
                # Create a donut chart
                wedges, texts, autotexts = ax.pie(
                    mood_df['Count'], labels=mood_df['Mood'], autopct='%1.1f%%', 
                    colors=pie_colors, startangle=90, pctdistance=0.85
                )
                
                # Draw a white circle in the middle to make it a donut
                centre_circle = plt.Circle((0,0),0.70,fc='#0E1117') # Match this to your background color
                fig.gca().add_artist(centre_circle)
                
                fig.patch.set_alpha(0.0)
                st.pyplot(fig)

            # 2. Monthly Sentiment Trend (Line Chart)
            with col_s2:
                st.caption("Average Sentiment Over Time (Monthly)")
                sentiment_timeline = helper.monthly_sentiment(selected_user, df)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(sentiment_timeline['time'], sentiment_timeline['compound_score'], color='#3B82F6', marker='o')
                
                # Add a horizontal line at 0 to clearly show positive vs negative territory
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                
                plt.xticks(rotation=45)
                ax.set_ylabel("Sentiment Score (-1 to 1)")
                fig.patch.set_alpha(0.0)
                st.pyplot(fig)

            st.markdown("---")
            col_s3, col_s4 = st.columns(2)

            # 3. Hourly Sentiment (Bar Chart)
            with col_s3:
                st.caption("Average Sentiment by Hour of Day")
                hourly_sent = helper.hourly_sentiment(selected_user, df)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                # Map colors based on positive (green) or negative (red) score
                bar_colors = ['#25D366' if score > 0 else '#EF4444' for score in hourly_sent['compound_score']]
                
                ax.bar(hourly_sent['hour'], hourly_sent['compound_score'], color=bar_colors)
                ax.axhline(0, color='gray', linestyle='-', linewidth=1)
                ax.set_xlabel("Hour of Day (0-23)")
                ax.set_ylabel("Sentiment Score")
                fig.patch.set_alpha(0.0)
                st.pyplot(fig)

            # 4. User Sentiment Ranking (Only for Overall)
            with col_s4:
                if selected_user == 'Overall':
                    st.caption("Most Positive vs. Most Negative Users")
                    user_rankings = helper.user_sentiment_ranking(df)
                    
                    # Take top 5 most negative and top 5 most positive to avoid crowding
                    if len(user_rankings) > 10:
                        user_rankings = pd.concat([user_rankings.head(5), user_rankings.tail(5)])
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bar_colors = ['#25D366' if score > 0 else '#EF4444' for score in user_rankings['compound_score']]
                    
                    # Horizontal bar chart for better name readability
                    ax.barh(user_rankings['users'], user_rankings['compound_score'], color=bar_colors)
                    ax.axvline(0, color='gray', linestyle='-', linewidth=1)
                    ax.set_xlabel("Average Sentiment Score")
                    fig.patch.set_alpha(0.0)
                    st.pyplot(fig)
                else:
                    st.info("💡 Select 'Overall' in the sidebar to see how users rank against each other in sentiment.")


        with tab6:
            st.subheader("✨ Generative AI Chat Insights")
            st.markdown("Powered by Groq & LLaMA-3")
            
            if not api_key:
                st.warning("🔑 Please enter a Groq API Key in the sidebar to unlock Generative AI features.")
            else:
                if selected_user == 'Overall':
                    st.markdown("### 🏆 Group Analysis Mode")
                    # Let the user choose the AI flavor
                    ai_mode = st.radio("Select mode:", ["Topic Analysis 📊", "Group Superlatives 🎭"], horizontal=True, label_visibility="collapsed")
                    
                    if st.button(f"Generate {ai_mode.split()[0]}", type="primary"):
                        st.session_state['show_llm_analysis'] = True
                        st.session_state['llm_mode'] = ai_mode # Save which mode they clicked!
                        
                    if st.session_state.get('show_llm_analysis', False) and st.session_state.get('llm_mode') == ai_mode:
                        with st.spinner("LLaMA-3 is analyzing the group dynamics..."):
                            
                            chat_sample = llm_helper.sample_chat_for_llm(df, selected_user)
                            
                            if "Topic" in ai_mode:
                                # --- 1. EXTRACT OVERALL GROUP STATS ---
                                total_msgs = len(df)
                                
                                # Get the top 3 most active users
                                active_users_df, _ = helper.fetch_frequent_users(df)
                                top_users = ", ".join(active_users_df['users'].head(3).tolist()) if not active_users_df.empty else "Unknown"
                                
                                # Get the top 10 most common words (excluding stopwords)
                                common_words_df = helper.most_common_words(selected_user, df)
                                # Assuming common_words_df has the words in the first column (index 0)
                                top_words = ", ".join(common_words_df[0].head(10).astype(str).tolist()) if not common_words_df.empty else "Unknown"
                                
                                group_stats_context = f"""
                                - Total Messages in Chat: {total_msgs}
                                - Most Active Members driving the conversation: {top_users}
                                - Top 10 Most Used Words (excluding standard grammar): {top_words}
                                """

                                # --- 2. CALL THE UPGRADED API ---
                                result = llm_helper.get_group_topics(chat_sample, group_stats_context, api_key)
                                
                                # --- 3. RENDER RESULTS ---
                                if not isinstance(result, dict) or "error" in result:
                                    st.error(f"Execution Error: {result.get('error', 'Unknown')}")
                                else:
                                    st.success("Analysis Complete!")
                                    st.write(f"**Group Dynamic:** {result.get('summary', '')}")
                                    
                                    st.markdown("### Top Discussion Themes")
                                    for topic in result.get('topics', []):
                                        if isinstance(topic, dict):
                                            st.markdown(f"**{topic.get('name', 'Unknown')}**")
                                            try:
                                                pct = min(1.0, max(0.0, float(topic.get('percentage', 0)) / 100.0))
                                                st.progress(pct)
                                            except Exception:
                                                st.write(f"Share: {topic.get('percentage')}%")
                                            st.caption(topic.get('description', ''))
                                            
                            elif "Superlative" in ai_mode:
                                result = llm_helper.get_group_superlatives(chat_sample, api_key)
                                if not isinstance(result, dict) or "error" in result:
                                    st.error(f"Execution Error: {result.get('error', 'Unknown')}")
                                else:
                                    st.success("The Yearbook is ready! 📸")
                                    superlatives = result.get('superlatives', [])
                                    
                                    if isinstance(superlatives, list):
                                        for item in superlatives:
                                            if isinstance(item, dict):
                                                # Use st.info to make each superlative look like a neat card
                                                st.info(f"**{item.get('user', 'Unknown')}** - 🏆 {item.get('title', 'Participant')}")
                                                st.write(item.get('reason', ''))
                                    else:
                                        st.warning("Could not render superlatives. Model returned an unexpected structure.")
                                        st.json(result)

                else:
                    st.markdown(f"### 👤 Analyze {selected_user}")
                    ai_mode = st.radio("Select mode:", ["Psychological Profile 🧠", "Roast My Texting 🔥"], horizontal=True, label_visibility="collapsed")
                    
                    if st.button(f"Generate {ai_mode.split()[0]}", type="primary"):
                        st.session_state['show_llm_analysis'] = True
                        st.session_state['llm_mode'] = ai_mode
                        
                    if st.session_state.get('show_llm_analysis', False) and st.session_state.get('llm_mode') == ai_mode:
                        with st.spinner(f"LLaMA-3 is analyzing {selected_user}'s data..."):

                            chat_sample = llm_helper.sample_chat_for_llm(df, selected_user)
                            
                            # --- 1. EXTRACT BEHAVIORAL METRICS ---
                            # Get the behavioral data for the whole chat
                            starters, killers, _, response_times = helper.behavioral_analysis(df)
                            
                            # Safely extract this specific user's metrics
                            user_starter = starters[starters['User'] == selected_user]
                            initiation_rate = user_starter['Starter Rate (%)'].values[0] if not user_starter.empty else "0"
                            
                            user_killer = killers[killers['User'] == selected_user]
                            kill_rate = user_killer['Killer Rate (%)'].values[0] if not user_killer.empty else "0"
                            
                            user_response = response_times[response_times['User'] == selected_user]
                            avg_resp = user_response['Avg Response Time (Mins)'].values[0] if not user_response.empty else "Unknown"
                            if avg_resp != "Unknown":
                                avg_resp = round(avg_resp, 1)

                            # Get basic stats and sentiment
                            user_df = df[df['users'] == selected_user]
                            total_msgs = len(user_df)
                            avg_sentiment = "Neutral"
                            if 'compound_score' in user_df.columns:
                                score = user_df['compound_score'].mean()
                                avg_sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                            most_active = f"{user_df['hour'].mode().iloc[0]}:00" if not user_df.empty else 'Unknown'

                            # --- 2. BUILD THE ULTIMATE STATS CONTEXT ---
                            stats_context = f"""
                            - Total Messages Sent: {total_msgs}
                            - Average Emotional Tone: {avg_sentiment}
                            - Most Active Time of Day: {most_active}
                            - Conversation Initiation Rate: {initiation_rate}% (How often they start a chat when present)
                            - Vibe Kill Rate: {kill_rate}% (How often they send the final message that no one replies to)
                            - Average Response Time: {avg_resp} minutes
                            """
                            
                            # --- 3. ROUTE TO THE CORRECT API ---
                            if "Profile" in ai_mode:
                                result = llm_helper.get_user_persona(chat_sample, selected_user, stats_context, api_key)
                                
                                if not isinstance(result, dict) or "error" in result:
                                    st.error(f"API Error: {result.get('error', 'Unknown')}")
                                else:
                                    st.success("Deep Behavioral Profile Generated!")
                                    st.markdown(f"### 🧠 The Archetype: {result.get('archetype', 'Unknown')}")
                                    traits = " • ".join(result.get('core_traits', []))
                                    st.caption(f"**Core Traits:** {traits}")
                                    st.markdown("---")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**🎭 Social Role in Group**")
                                        st.info(result.get('social_role', 'N/A'))
                                        st.markdown("**🗣️ Communication Style**")
                                        st.info(result.get('communication_style', 'N/A'))
                                    with col2:
                                        st.markdown("**🎯 Top Topics of Interest**")
                                        for topic in result.get('top_interests', []):
                                            st.markdown(f"- {topic}")
                                            
                            elif "Roast" in ai_mode:
                                result = llm_helper.get_user_roast(chat_sample, selected_user, stats_context, api_key)
                                
                                if not isinstance(result, dict) or "error" in result:
                                    st.error(f"API Error: {result.get('error', 'Unknown')}")
                                else:
                                    st.success("Boom. Roasted. 🎤")
                                    st.markdown(f"### 🔥 The Roast of {selected_user}")
                                    st.write(result.get('brutal_roast', 'Model refused to roast.'))
                                    
                                    st.markdown("---")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.error(f"🚩 **Biggest Red Flag:**\n\n{result.get('biggest_red_flag', 'N/A')}")
                                    with col2:
                                        st.success(f"🟩 **Biggest Green Flag:**\n\n{result.get('biggest_green_flag', 'N/A')}")

else:
    # A welcoming empty state
    st.info("👈 Please upload a WhatsApp chat export to begin analysis.")