import streamlit as st
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

# Load vectorizer once
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Dictionary mapping model names to files
models = {
    "Logistic Regression": "LogisticRegression_model.pkl",
    "SVM": "SVM_model.pkl",
    "Random Forest": "RandomForest_model.pkl",
    "Decision Tree": "DecisionTree_model.pkl"
}

# PDF generation using reportlab
def generate_pdf(text, model_name, prediction, filename="result.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "Sentiment Analysis Result")

    # Model used & prediction
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Model Used: {model_name}")
    c.drawString(50, height - 120, f"Predicted Sentiment: {prediction}")

    # Input text (multi-line support)
    text_lines = text.split("\n")
    y = height - 160
    for line in text_lines:
        c.drawString(50, y, line)
        y -= 20
        if y < 50:  # if page ends, add new page
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50

    c.save()
    return filename

# Streamlit UI
st.title("Sentiment Analysis Deployment")
model_choice = st.selectbox("Select Model", list(models.keys()))
user_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Predict"):
    # Load the selected model
    with open(models[model_choice], 'rb') as f:
        model = pickle.load(f)

    # Vectorize input text
    text_vec = vectorizer.transform([user_text.lower()])

    # Predict sentiment
    pred = model.predict(text_vec)[0]

    st.write(f"**Predicted Sentiment:** {pred}")

    # Visualize sentiment with a simple bar chart (example data)
    labels = ['Positive', 'Negative', 'Neutral']
    # Dummy values for visualization; replace with real model outputs if available
    values = [0, 0, 0]
    if pred.lower() == 'positive':
        values = [1, 0, 0]
    elif pred.lower() == 'negative':
        values = [0, 1, 0]
    else:
        values = [0, 0, 1]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['green', 'red', 'gray'])
    ax.set_title('Sentiment Prediction Visualization')
    st.pyplot(fig)

    # Generate PDF and enable download
    pdf_path = generate_pdf(user_text, model_choice, pred)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download Result as PDF",
            data=f,
            file_name=pdf_path,
            mime="application/pdf"
        )
