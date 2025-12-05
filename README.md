Skin Cancer Classifier + Grad-CAM
--------------------------------

Files:
- app.py
- gradcam.py
- helper.py
- requirements.txt
- skin_cancer_model.h5  <-- add your .h5 model here (saved from Colab)

How to run locally:
1. create virtualenv
2. pip install -r requirements.txt
3. streamlit run app.py

Deploy:
1. Push repo to GitHub (use Git LFS for the .h5 if >25MB)
2. On Streamlit Cloud, connect repo and select app.py
