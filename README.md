# Smart Recruitment Assistant

Predictive analytics for employee job-change risk (ITI graduation project).
A Streamlit interface around the original trained models.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Structure
- `app.py` – Streamlit entry point
- `recruitment_pipeline.py` – preprocessing that reproduces the notebook (cleaning, split, scaler/encoder) and model loading
- `models/` – original pickled models (Gradient Boosting is the final model)
- `data/aug_train.csv` – dataset (needed at runtime to rebuild the preprocessor)
- `notebooks/smart_jop.ipynb` – original training notebook (unchanged)
