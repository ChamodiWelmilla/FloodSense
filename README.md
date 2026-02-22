# 🌊 FloodSense

AI-powered flood risk assessment for Sri Lanka using machine learning.

## Overview

This application predicts flood probability using a Random Forest model trained on Sri Lankan flood data. It analyzes 62 environmental factors to provide real-time risk assessment.

## Features

- **25 Sri Lankan Districts** with pre-configured data
- **Risk Levels**: Low (<30%), Moderate (30-60%), Critical (>60%)
- **Customizable Inputs**: Rainfall, river distance, elevation, land type
- **Interactive Map** with location visualization
- **98.45% Accuracy** across test scenarios

## Installation

```bash
pip install streamlit pandas scikit-learn joblib numpy
```

## Usage

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### How to Use

1. **Select District** from sidebar
2. **Set Rainfall** (7-day cumulative in mm)
3. **Adjust Distance** to nearest river
4. **Customize Location** (optional) - uncheck "Use District Location"
5. View **Risk Prediction** and recommendations

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 62 total (9 numeric + 53 categorical)
- **Key Inputs**: Rainfall, elevation, soil type, land cover, population density
- **Performance**: 98.45% success rate

## Supported Districts

Ampara, Anuradhapura, Badulla, Batticaloa, Colombo, Galle, Gampaha, Hambantota, Jaffna, Kalutara, Kandy, Kegalle, Kilinochchi, Kurunegala, Mannar, Matale, Matara, Monaragala, Mullaitivu, Nuwara Eliya, Polonnaruwa, Puttalam, Ratnapura, Trincomalee, Vavuniya

## Files

- `app.py` - Main application
- `flood_model.pkl` - Trained ML model
- `scaler.pkl` - Feature scaler
- `feature_columns.pkl` - Feature definitions

## License

Provided as-is for flood risk assessment and disaster management.

---

⚡ Powered by Random Forest ML Model | Based on 62 environmental factors

