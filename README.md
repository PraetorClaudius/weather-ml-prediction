# Weather Temperature Prediction System

An end-to-end machine learning system that predicts tomorrow's temperature for Mexican cities with 2.6°C accuracy using automated data collection and deployment.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Try it now:** [Weather Temperature Predictor](https://weather-ml-prediction-shidq7bz8ukz5rx2zqoxax.streamlit.app)

## Project Highlights

 **Automated Data Pipeline** - AWS Lambda collects weather data every 6 hours  
 **Real-World Data** - 550+ records from 5 Mexican cities over 3+ weeks  
 **High Accuracy** - Mean Absolute Error of 2.19°C (R² = 0.747)  
 **Production Deployment** - Live web app on Streamlit Cloud  
 **Full ML Workflow** - Data collection → Feature engineering → Training → Deployment  

---

## Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Absolute Error (MAE)** | 2.19°C | Predictions within ~2.19 degrees on average |
| **R² Score** | 0.747 | Model explains 74.7% of temperature variance |
| **Best Algorithm** | Lasso Regression | Simple, interpretable, generalizes well |
| **Training Data** | 550+ records | 3+ weeks of continuous collection |

**Performance Context:**  
Professional weather forecasting typically achieves 2-3°C accuracy for 24-hour predictions. Our model performs at a respectable, real-world level.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

OpenWeatherMap API (every 6h)
         ↓
    AWS Lambda (Collector)
         ↓
    AWS S3 (Raw Data)
         ↓
    AWS Lambda (Processor) ← batch_process.py
         ↓
    AWS S3 (Processed Data)
         ↓
    Local: Feature Engineering → Model Training → Evaluation
         ↓
    Streamlit Web App (Deployed on Cloud)
```

### Data Flow

1. **Collection (Automated)**: EventBridge triggers Lambda every 6 hours → Calls OpenWeatherMap API → Stores raw JSON in S3
2. **Processing**: Batch script cleans data → Extracts relevant fields → Saves to processed bucket
3. **Feature Engineering**: Creates time-series features (lags, rolling stats, temporal patterns)
4. **Model Training**: Tests 6 algorithms → Selects best performer (Lasso) → Saves artifacts
5. **Deployment**: Streamlit app loads model → Users input conditions → Returns 24h prediction

---

## Technologies & Tools

### Data Pipeline
- **AWS Lambda** - Serverless data collection & processing
- **AWS S3** - Scalable data storage
- **AWS EventBridge** - Scheduled automation
- **OpenWeatherMap API** - Real-time weather data source

### Machine Learning
- **Python 3.12** - Primary language
- **scikit-learn** - ML algorithms (Lasso, Random Forest, Gradient Boosting)
- **pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

### Deployment
- **Streamlit** - Interactive web application
- **Plotly** - Dynamic charts
- **Streamlit Cloud** - Free hosting

---

## Project Structure

```
weather-ml-prediction/
├── data/
│   ├── downloaded/           # Raw data from S3
│   ├── processed/            # Featured data for ML
│   │   └── featured_data.csv
│   └── models/               # Trained model artifacts
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── feature_names.pkl
│       └── model_metadata.json
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── app/
│   └── streamlit_app.py      # Web application
├── batch_process.py          # S3 data processor
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- AWS account (for data collection)
- OpenWeatherMap API key (free tier)

### Installation

1. **Clone repository:**
```bash
git clone https://github.com/PraetorClaudius/weather-ml-prediction.git
cd weather-ml-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
cd app
python -m streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## How It Works

### Feature Engineering

The model uses sophisticated time-series features:

**Lag Features** (Previous measurements)
- Temperature 6, 12, and 24 hours ago
- Previous humidity and pressure readings

**Rolling Statistics** (Trends)
- 24-hour average, min, max temperature
- Temperature volatility (standard deviation)

**Temporal Features**
- Hour of day, day of week, month
- Weekend indicator

**Weather Conditions**
- Humidity, pressure, wind speed, cloudiness

### Model Selection

Tested 6 algorithms:
| Model | Test MAE | Test R² |
|-------|----------|---------|
| **Lasso Regression** | **2.19°C** | **0.747** |
| Random Forest | 2.78°C | 0.698 |
| Gradient Boosting | 2.85°C | 0.681 |
| Ridge Regression | 2.91°C | 0.665 |
| Linear Regression | 3.02°C | 0.642 |
| Decision Tree | 3.45°C | 0.587 |

**Winner:** Lasso Regression - Best balance of accuracy, simplicity, and generalization.

---

## Results & Insights

### Key Findings

1. **Temperature Patterns are Predictable**: R² of 0.747 shows strong correlation between features and next-day temperature
2. **Recent Temps Matter Most**: Lag features (6-24h ago) were the most important predictors
3. **Time of Day Influences**: Hour and day-of-week features improved accuracy by ~15%
4. **Simple Models Win**: Lasso outperformed complex ensemble methods, proving that regularization > complexity

### Model Limitations

- **Geographic Scope**: Currently limited to 5 Mexican cities
- **Prediction Window**: Only predicts 24 hours ahead (not multi-day forecasts)
- **Weather Events**: May struggle with extreme/rare weather events (limited training data)
- **Data Volume**: More historical data (6+ months) would likely improve accuracy

---

## What I Learned

### Technical Skills Gained
- Building production ML pipelines from scratch
- Working with time-series data and feature engineering
- Model selection, evaluation, and hyperparameter tuning
- Cloud deployment with AWS (Lambda, S3, EventBridge)
- Creating interactive web applications with Streamlit

### Data Science Best Practices
- Proper train/test splitting for time-series (avoiding data leakage)
- Feature engineering dramatically impacts performance
- Simpler models often generalize better than complex ones
- Documentation and reproducibility are crucial

### Engineering Principles
- Automated data collection reduces manual work
- Serverless architecture minimizes costs (100% free tier!)
- Good error handling and logging saves debugging time
- Version control and modular code structure matter

---

## Future Enhancements

### Planned Improvements

**Short-term (1-2 months)**
- [ ] Expand to 20+ cities across Latin America
- [ ] Add LSTM/GRU models for better time-series predictions
- [ ] Implement automated retraining pipeline
- [ ] Add confidence intervals to predictions

**Long-term (6+ months)**
- [ ] Multi-day forecasts (3-7 days ahead)
- [ ] Mobile app version (React Native)
- [ ] Email/SMS alerts for extreme weather
- [ ] Real-time model performance monitoring
- [ ] API endpoint for programmatic access

---

## Development

### Running Notebooks

Explore the analysis step-by-step:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Training a New Model

1. Collect data (let Lambda run for 3+ weeks)
2. Process raw data: `python batch_process.py`
3. Run feature engineering notebook
4. Run model training notebook
5. Update app with new model artifacts

### Deployment

The app is deployed on Streamlit Cloud. To deploy your own:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set main file: `app/streamlit_app.py`
5. Deploy!

---

## Data Sources

- **Primary**: [OpenWeatherMap API 2.5](https://openweathermap.org/api) (Free tier)
- **Cities**: Toluca, Mexico City, Guadalajara, Monterrey, Cancún
- **Collection Frequency**: Every 6 hours (4 times daily)
- **Data Retention**: Indefinite (stored in AWS S3)

---

## Author

**Eduardo Arriaga Alejandre**

Telematics Engineer transitioning to Data Science & ML Engineering

- 🔗 [LinkedIn](https://www.linkedin.com/in/eduardo-arriaga-230156295/) ← Update this
- 💻 [GitHub](https://github.com/PraetorClaudius)
- 📧 [Email](earriaga0226@gmail.com) ← Update this

### Other Projects

- [AWS Weather Data Pipeline](https://github.com/PraetorClaudius/aws-weather-data-pipeline/tree/main) - The data collection system powering this project

---

## Acknowledgments

- **OpenWeatherMap** for providing free API access
- **Streamlit** for free hosting and excellent framework
- **AWS Free Tier** for serverless infrastructure
- **scikit-learn** community for comprehensive ML tools
