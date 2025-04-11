import streamlit as st
from engine import prediction
from classes import disease_class
from PIL import Image
import requests
import json
from dotenv import load_dotenv
import os
from langchain_cohere import ChatCohere
from crewai import Crew, Process
from Agent.agents import DiseasePredictorAgent
from Agent.tasks import DiseasePredictorTask
from crewai import LLM

load_dotenv()

GEMINI_API_KEY = "AIzaSyAoXkCAtHMMxphFRSHP1Wc78asPiY-pYC4"
# os.environ['OPENAI_API_KEY'] = "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"


def predict(uploaded_file):
    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        class_index, _ = prediction(uploaded_file)
        label = disease_class[class_index]
        return image, class_index, label
    else:
        return None, None, None

load_dotenv()
api_key = os.getenv('COHERE_API_KEY')

def DiseasePredictor(llm, disease):
    # Instantiate the agent and task classes properly
    agent = DiseasePredictorAgent()
    task = DiseasePredictorTask()

    # Create the disease researcher agent and task for the specified disease
    researcher_agent = agent.create_disease_researcher(llm, disease)
    researcher_task = task.create_disease_researcher_task(researcher_agent, disease)

    # Forming the crew configuration with enhanced configurations
    crew = Crew(
        agents=[researcher_agent],  
        tasks=[researcher_task],   
        process=Process.sequential,  
    )

    # Start the task execution process and get feedback
    result = crew.kickoff()  # This should execute the tasks in order
    return result

def main():
    st.title('Disease Detection App')
    llm = LLM(
        model="gemini/gemini-1.5-pro-latest",
        temperature=0.7,
        api_key=GEMINI_API_KEY
    )
    uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])
    image, class_index, label = predict(uploaded_file)

    if uploaded_file is not None and label:
        st.image(image)
        st.write("Prediction : ", label)
        st.divider()

    if label and st.button('Get Information'):
        result = DiseasePredictor(llm, label)
        st.write(result.raw)



if __name__ == '__main__':
    main()