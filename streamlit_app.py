import streamlit as st
import pandas as pd
from pycaret.regression import predict_model, load_model
import numpy as np


st.set_page_config(layout="wide")

st.title("Match Demo")
st.subheader("Add your inputs")
col1, col2, col3 = st.columns(3, gap="small")
with col1: 
    a = st.number_input('T Cell-to-Cancer Cell Ratio (#)')
    d = st.number_input('CD3 Molecules per T-Cell')
    g = st.number_input('CD38 Molecules per Cancer Cell')
with col2:
    b = st.number_input('Mean T-Cell Count (total # per well)')
    e = st.number_input('CD19 Molecules per Cancer Cell')
with col3: 
    c = st.number_input('Mean Cancer-Cell Count (total # per well)')
    f = st.number_input('CD20 Molecules per Cancer Cell')

if st.button("Predict best dose for minimal cancer cells"):
    model_cancer = load_model('models/xgboost')
    model_tcell= load_model('models/xgboost_tcell')
    test_d = []
    for w in [0,1.66, 5, 15]:
        for x in [0,0.833, 1.66, 2.5, 3.33, 4.16, 5]:
            for y in [0,0.833, 1.66, 2.5, 3.33, 4.16, 5]:
                for z in [0,0.833, 1.66, 2.5, 3.33, 4.16, 5]:
                    test_d.append([a,b,c,d,e,f,g,w,x,y,z])
    test_data = pd.DataFrame(test_d,
                        columns=['T Cell-to-Cancer Cell Ratio (#)','Mean T-Cell Count (total # per well)',
        'Mean Cancer-Cell Count (total # per well)', 'CD3 Molecules per T-Cell','CD19 Molecules per Cancer Cell', 
        'CD20 Molecules per Cancer Cell', 'CD38 Molecules per Cancer Cell', '[Fab\'CD3-MORF2] (nM)',
        '[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)','[Fab\'CD38-MORF1] (nM)',])
    pred_cancer = predict_model(model_cancer, data=test_data)
    pred_cancer = pred_cancer.rename(columns={"prediction_label":"prediction cancer cells"})
    best_cancer = pred_cancer[pred_cancer["prediction cancer cells"] == pred_cancer["prediction cancer cells"].min()]
    pred_tcell = predict_model(model_tcell, data=best_cancer.drop(columns=["prediction cancer cells"]))
    best_cancer["prediction t-cells"] = pred_tcell["prediction_label"]
    st.write(best_cancer[['[Fab\'CD3-MORF2] (nM)','[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)','[Fab\'CD38-MORF1] (nM)',
                          "prediction cancer cells", "prediction t-cells"]].sort_values(by="prediction t-cells"))

    test_d = []
    for w in [0,1.66, 5]:
        for x in [0,0.833, 1.66, 2.5, 3.33, 4.16, 5]:
            for y in [0,0.833, 1.66, 2.5, 3.33, 4.16, 5]:
                for z in [0,0.833, 1.66, 2.5, 3.33, 4.16, 5]:
                    test_d.append([a,b,c,d,e,f,g,w,x,y,z])
    test_data = pd.DataFrame(test_d,
                        columns=['T Cell-to-Cancer Cell Ratio (#)','Mean T-Cell Count (total # per well)',
        'Mean Cancer-Cell Count (total # per well)', 'CD3 Molecules per T-Cell','CD19 Molecules per Cancer Cell', 
        'CD20 Molecules per Cancer Cell', 'CD38 Molecules per Cancer Cell', '[Fab\'CD3-MORF2] (nM)',
        '[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)','[Fab\'CD38-MORF1] (nM)',])
    pred_cancer = predict_model(model_cancer, data=test_data)
    pred_cancer = pred_cancer.rename(columns={"prediction_label":"prediction cancer cells"})
    best_cancer = pred_cancer[pred_cancer["prediction cancer cells"] == pred_cancer["prediction cancer cells"].min()]
    pred_tcell = predict_model(model_tcell, data=best_cancer.drop(columns=["prediction cancer cells"]))
    best_cancer["prediction t-cells"] = pred_tcell["prediction_label"]
    st.write(best_cancer[['[Fab\'CD3-MORF2] (nM)','[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)','[Fab\'CD38-MORF1] (nM)',
                          "prediction cancer cells", "prediction t-cells"]].sort_values(by="prediction t-cells"))