import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainable_chatbot import ExplainableChatbot

def run_customer_service_example():
    """
    Example demonstrating explainable chatbot for customer service scenarios.
    """
    # Initialize chatbot
    chatbot = ExplainableChatbot(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        num_labels=3
    )
    
    # Example customer service interactions
    interactions = [
        "The product arrived damaged and customer service was unhelpful.",
        "The representative was polite but couldn't solve my problem.",
        "Outstanding service! They went above and beyond to help me.",
        "I had to wait for 30 minutes before speaking to someone.",
        "The issue was resolved quickly and professionally."
    ]
    
    # Process each interaction and get explanations
    for i, text in enumerate(interactions, 1):
        print(f"\nAnalyzing Interaction #{i}")
        print("-" * 50)
        print(f"Customer: {text}")
        
        # Get explanations
        explanations = chatbot.get_complete_explanation(text)
        
        # Save explanations to a directory for this interaction
        output_dir = f"explanations/interaction_{i}"
        chatbot.save_explanations(explanations, output_dir)
        
        # Print results
        prediction = explanations["prediction"]
        print(f"\nSentiment: {prediction['class']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"\nExplanations saved to: {output_dir}")
        print("\nKey factors in the decision:")
        
        # Show LIME top features
        lime_exp = explanations["explanations"]["lime"]
        if "local_exp" in lime_exp:
            features = lime_exp["local_exp"][1]  # Get features for predicted class
            print("\nLIME Analysis:")
            for feature, weight in sorted(features, key=lambda x: abs(x[1]), reverse=True)[:3]:
                print(f"- Feature: {feature}, Impact: {weight:.3f}")
        
        print("-" * 50)

if __name__ == "__main__":
    run_customer_service_example()