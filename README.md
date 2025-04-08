# Explainable AI Chatbot

A comprehensive implementation of an explainable AI chatbot that provides transparency in its decision-making process using SHAP, LIME, and GradCAM. This project is particularly useful for customer service bots and educational tutors where understanding the reasoning behind responses is crucial.

## 🌟 Features

### 1. Multiple Explainability Methods
- **SHAP (SHapley Additive exPlanations)**
  - Feature attribution analysis
  - Global and local interpretability
  - Visual explanations of model decisions

- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Local feature importance
  - Text snippet highlighting
  - Interpretable representation

- **GradCAM (Gradient-weighted Class Activation Mapping)**
  - Attention visualization
  - Neural network interpretation
  - Visual heatmaps of decision focus

### 2. Use Cases
- **Customer Service**
  - Sentiment analysis explanation
  - Response reasoning
  - Decision confidence metrics

- **Educational Tutoring**
  - Answer explanation
  - Concept relevance
  - Learning path recommendations

### 3. Visualization
- Interactive plots
- Attention heatmaps
- Feature importance graphs
- Decision confidence metrics

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+ required
python --version
```

### Installation
```bash
# Clone the repository
git clone https://github.com/whitecloud343/explainable-ai-chatbot.git
cd explainable-ai-chatbot

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
from src.explainable_chatbot import ExplainableChatbot

# Initialize chatbot
chatbot = ExplainableChatbot()

# Get explanations for a response
text = "I really enjoyed the customer service experience!"
explanations = chatbot.get_complete_explanation(text)

# Save explanations and visualizations
chatbot.save_explanations(explanations)
```

## 📊 Example Outputs

### SHAP Analysis
```
Text: "The customer service was excellent"
Top Contributing Words:
1. excellent (+0.82)
2. service (+0.31)
3. customer (+0.15)
```

### LIME Explanation
```
Highlighted Text:
"The [customer service](+0.45) was [excellent](+0.82)"
```

### GradCAM Visualization
- Generates attention heatmaps showing which parts of the input influenced the decision most strongly

## 🛠️ Technical Architecture

### Components

1. **Core Chatbot**
   - Transformer-based model
   - Multi-class classification
   - Confidence scoring

2. **Explainability Layer**
   - SHAP analyzer
   - LIME interpreter
   - GradCAM visualizer

3. **Visualization Engine**
   - Interactive plots
   - Heatmaps
   - Feature importance graphs

## 📈 Use Cases

### 1. Customer Service Bot
```python
# Example from examples/customer_service_example.py
from examples.customer_service_example import run_customer_service_example

run_customer_service_example()
```

### 2. Educational Tutor
```python
# Coming soon: Educational tutor example
```

## 📊 Performance Metrics

- Response accuracy
- Explanation quality
- Processing time
- Memory usage
- GPU utilization

## 🔧 Configuration

Key parameters that can be configured:
- Model selection
- Number of classes
- Explanation detail level
- Visualization options
- Output formats

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dev dependencies
pip install -r requirements-dev.txt
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Examples

### Running the Customer Service Example
```bash
python examples/customer_service_example.py
```

This will:
1. Process sample customer interactions
2. Generate explanations using all three methods
3. Save visualizations and explanations
4. Print analysis results

## 📚 Documentation

Full documentation is available in the `docs` directory:
- API Reference
- Usage Examples
- Configuration Guide
- Best Practices

## 🔄 Updates and Versioning

- Version: 1.0.0
- Last Updated: April 2025
- Python: 3.8+
- Dependencies: See requirements.txt

## 📞 Support

For issues and questions:
- Create an issue in this repository
- Check existing documentation
- Review closed issues for solutions

## 🎯 Future Enhancements

1. Additional explainability methods
2. More use case examples
3. Enhanced visualization options
4. Performance optimizations
5. Additional model support

## 🏆 Acknowledgments

- SHAP library developers
- LIME project contributors
- GradCAM implementation team
- Hugging Face Transformers team