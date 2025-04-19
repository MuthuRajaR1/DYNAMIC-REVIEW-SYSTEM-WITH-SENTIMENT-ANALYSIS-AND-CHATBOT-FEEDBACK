import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import matplotlib.pyplot as plt

# Download NLTK VADER lexicon (needed only once)
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

st.session_state.review_history = []

# Function to analyze sentiment
def analyze_sentiment(review):
    sentiment = sia.polarity_scores(str(review))  # Convert review to string
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to generate chatbot suggestion
def generate_suggestion(sentiment):
    suggestions = {
        "Positive": [
            "Thank you for your kind words! We're so happy you had a great experience.",
            "That’s wonderful to hear! Your support means the world to us.",
            "We’re thrilled you loved it! Is there anything else we can do to make it even better?",
            "Your feedback made our day! Would you like to share your experience with others?",
            "We truly appreciate your support! Looking forward to serving you again soon.",
            "So glad to hear that! If you ever need anything, we're here for you.",
            "It’s always a pleasure to provide great service! Let us know how we can continue to impress you.",
            "Your review just made us smile! Thank you for being an amazing customer.",
            "Hearing this makes all our hard work worthwhile! Hope to see you again soon.",
            "We’re beyond grateful for your support! Would you recommend us to your friends?"
        ],
        "Negative": [
            "We're sorry to hear that you had a bad experience. Can you tell us more so we can make things right?",
            "We truly value your feedback and apologize for any inconvenience. How can we improve?",
            "Your concerns matter to us. Let’s work together to find a solution that makes things better for you.",
            "We appreciate your honesty and want to make things right. Can you share more details with us?",
            "We’re really sorry to hear this. Please let us know how we can fix it and improve your experience.",
            "That’s not the experience we want you to have. Would you like to speak with our support team?",
            "We take your feedback seriously and are always looking to improve. Thank you for sharing.",
            "Sorry for the inconvenience. We’d love another chance to serve you better next time!",
            "We’re disappointed to hear this. Please let us know how we can make it up to you.",
            "Your feedback is important, and we’re committed to improving. How can we make it right?"
        ],
        "Neutral": [
            "Thank you for your feedback! Is there anything we can do to improve your experience?",
            "We appreciate your input. What features or improvements would you like to see?",
            "Your feedback helps us grow. Do you have any suggestions on how we can enhance our product/service?",
            "We’re glad to hear from you! What aspects of our service stood out to you the most?",
            "Thanks for sharing your thoughts! How can we make your experience even better?",
            "Your feedback is valuable to us. Is there anything specific you'd like us to change or add?",
            "We’re always looking to improve. What would make your next experience even better?",
            "Thank you for your review! Would you be open to sharing more details about your experience?",
            "Your thoughts matter to us! Let us know how we can make our service more useful for you.",
            "We appreciate your time in sharing feedback. Is there anything we could do differently next time?"
        ]
    }
    return random.choice(suggestions[sentiment])
# Function to plot sentiment distribution
def plot_sentiment_graphs(df):
    sentiment_counts = df['sentiment'].value_counts()
    
    # PIE CHART
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=['green', 'red', 'gold'], startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    st.subheader("📊 Sentiment Distribution (Pie Chart)")
    st.pyplot(fig1)

    # BAR CHART
    fig2, ax2 = plt.subplots()
    ax2.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gold'])
    ax2.set_xlabel("Sentiment")
    ax2.set_ylabel("Count")
    ax2.set_title("📈 Sentiment Distribution (Bar Chart)")
    st.pyplot(fig2)

# Streamlit UI
st.title("📊 Dynamic Review Sentiment Analysis with Chatbot Suggestions & Graphs")

# User chooses mode
option = st.radio("Choose an option:", ["Single Review Analysis", "CSV File Analysis"])

if option == "Single Review Analysis":
    review = st.text_area("Enter your review:")
    if st.button("Analyze"):
        if review.strip():
            sentiment = analyze_sentiment(review)
            suggestion = generate_suggestion(sentiment)
             
            # Store review history
            st.session_state.review_history.append({
                "Review": review, "Sentiment": sentiment, "Chatbot Suggestion": suggestion
            })

            st.subheader("🔍 Review Analysis:")
            st.write(f"**Review:** {review}")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Chatbot Suggestion:** {suggestion}")
        else:
            st.warning("⚠️ Please enter a review.")
    
     # Display history
    if st.session_state.review_history:
        st.write("### 📜 Review History")
        st.dataframe(pd.DataFrame(st.session_state.review_history))

elif option == "CSV File Analysis":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'review' in df.columns:
            df['sentiment'] = df['review'].apply(analyze_sentiment)
            df['chatbot_suggestion'] = df['sentiment'].apply(generate_suggestion)

            st.subheader("✅ Processed Data:")
            st.write(df[['review', 'sentiment', 'chatbot_suggestion']])

            # Plot graphs
            plot_sentiment_graphs(df)

            # Save the analyzed file
            df.to_csv("analyzed_reviews.csv", index=False)
            st.download_button("📥 Download Processed CSV", df.to_csv(index=False), file_name="analyzed_reviews.csv")
        else:
            st.error("⚠️ Error: The CSV file must contain a 'review' column.")
