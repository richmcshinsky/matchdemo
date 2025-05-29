import streamlit as st
import pandas as pd
from pycaret.regression import predict_model, load_model, setup, create_model, tune_model


st.set_page_config(layout="wide")

st.title("Match Demo")
st.divider()

st.subheader("Load Data and Train Model")
@st.cache_data
def load_data(cols):
    df = pd.read_excel("models/datafile.xlsx")
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    #df2 = pd.read_excel("models/datafile2.xlsx")
    #df2 = df2.T
    #df2.columns = df2.iloc[0]
    #df2 = df2.drop(df2.index[0])
    #df = pd.concat([df, df2])
    df = df[cols]
    #df = df.drop(columns=["Mean Residual T-Cell Count (total # per well)"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    df["ratio of cancer to t-cell"] = df["Mean Residual Cancer-Cell Count (total # per well)"]/df["Mean Residual T-Cell Count (total # per well)"]
    return df

@st.cache_data
def train_model(df, target, ignore):
    s = setup(df, target = target, train_size=0.90, session_id = 123,ignore_features=ignore)
    best = create_model('xgboost')
    tuned_best = tune_model(best)
    return tuned_best

columns = st.multiselect("Select input columns for training the model",
    ['T Cell-to-Cancer Cell Ratio (#)','Mean T-Cell Count (total # per well)', 'Mean Cancer-Cell Count (total # per well)', 
     'CD3 Molecules per T-Cell', 'CD19 Molecules per Cancer Cell', 'CD20 Molecules per Cancer Cell', 
     'CD38 Molecules per Cancer Cell', '[Fab\'CD3-MORF2] (nM)', '[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)', 
     '[Fab\'CD38-MORF1] (nM)'],
    default=['T Cell-to-Cancer Cell Ratio (#)','Mean T-Cell Count (total # per well)', 'Mean Cancer-Cell Count (total # per well)', 
     'CD3 Molecules per T-Cell', 'CD19 Molecules per Cancer Cell', 'CD20 Molecules per Cancer Cell', 
     'CD38 Molecules per Cancer Cell', '[Fab\'CD3-MORF2] (nM)', '[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)', 
     '[Fab\'CD38-MORF1] (nM)'])
df = load_data(columns + ["Mean Residual Cancer-Cell Count (total # per well)", "Mean Residual T-Cell Count (total # per well)"])
model_cancer = train_model(df, "Mean Residual Cancer-Cell Count (total # per well)", 
                           ["Mean Residual T-Cell Count (total # per well)", "ratio of cancer to t-cell"])
model_tcell = train_model(df, "Mean Residual T-Cell Count (total # per well)", 
                          ["Mean Residual Cancer-Cell Count (total # per well)", "ratio of cancer to t-cell"])
model_rec = train_model(df, "ratio of cancer to t-cell", 
                        ["Mean Residual T-Cell Count (total # per well)", "Mean Residual Cancer-Cell Count (total # per well)"])
st.divider()

st.subheader("Make a prediction")
col1, col2 = st.columns(2, gap="small")
with col1:
    st.write("Inputs")
    a = st.number_input('T Cell-to-Cancer Cell Ratio (#)') if 'T Cell-to-Cancer Cell Ratio (#)' in columns else None
    b = st.number_input('Mean T-Cell Count (total # per well)') if 'Mean T-Cell Count (total # per well)' in columns else None
    c = st.number_input('Mean Cancer-Cell Count (total # per well)') if 'Mean Cancer-Cell Count (total # per well)' in columns else None
    d = st.number_input('CD3 Molecules per T-Cell') if 'CD3 Molecules per T-Cell' in columns else None
    e = st.number_input('CD19 Molecules per Cancer Cell') if 'CD19 Molecules per Cancer Cell' in columns else None
    f = st.number_input('CD20 Molecules per Cancer Cell') if 'CD20 Molecules per Cancer Cell' in columns else None
    g = st.number_input('CD38 Molecules per Cancer Cell') if 'CD38 Molecules per Cancer Cell' in columns else None

with col2:
    st.write("Grid search dosage options")
    if ('[Fab\'CD19-MORF1] (nM)' in columns) and ('[Fab\'CD20-MORF1] (nM)' in columns) and ('[Fab\'CD38-MORF1] (nM)' in columns):
        dose_max = st.number_input("Max total dosage for cd19, cd20, and cd38", 5)
    elif ('[Fab\'CD19-MORF1] (nM)' in columns) and ('[Fab\'CD20-MORF1] (nM)' in columns):
        dose_max = st.number_input("Max total dosage for cd19 and cd20", 5)
    else:
        dose_max = 100
    if '[Fab\'CD3-MORF2] (nM)' in columns:
        w_opts = st.text_input("CD3 search options", "0,1.66,5,15")
        w_opts = w_opts.split(",")
        w_opts = [float(x) for x in w_opts]
    else:
        w_opts = None
    if '[Fab\'CD19-MORF1] (nM)' in columns:
        x_opts = st.text_input("CD19 search options", "0,0.833,1.66,2.5,3.33,4.16,5")
        x_opts = x_opts.split(",")
        x_opts = [float(x) for x in x_opts]
    else:
        x_opts = None
    if '[Fab\'CD20-MORF1] (nM)' in columns:
        y_opts = st.text_input("CD20 search options", "0,0.833,1.66,2.5,3.33,4.16,5")
        y_opts = y_opts.split(",")
        y_opts = [float(x) for x in y_opts]
    else:
        y_opts = None
    if '[Fab\'CD38-MORF1] (nM)' in columns:
        z_opts = st.text_input("CD38 search options", "0,0.833,1.66,2.5,3.33,4.16,5")
        z_opts = z_opts.split(",")
        z_opts = [float(x) for x in z_opts]
    else:
        z_opts = None

if st.button("Predict best dose for minimal cancer cells and best RECOMMENDED dose (cancer cells/t-cell ratio)"):
    test_d = []
    if w_opts and x_opts and y_opts and z_opts:
        for w in w_opts:
            for x in x_opts:
                for y in y_opts:
                    for z in z_opts:
                        if x+y+z <= dose_max: # max dose
                            test_d.append([x for x in [a,b,c,d,e,f,g] if x is not None] + [w,x,y,z])
        test_data = pd.DataFrame(test_d,columns=columns)
    elif w_opts and x_opts and y_opts:
        for w in w_opts:
            for x in x_opts:
                for y in y_opts:
                    if x+y <= dose_max: # max dose
                        test_d.append([x for x in [a,b,c,d,e,f,g] if x is not None] + [w,x,y])
        test_data = pd.DataFrame(test_d,columns=columns)
    pred_cancer = predict_model(model_cancer, data=test_data)
    pred_cancer = pred_cancer.rename(columns={"prediction_label":"prediction cancer cells"})
    best_cancer = pred_cancer[pred_cancer["prediction cancer cells"] == pred_cancer["prediction cancer cells"].min()]
    pred_tcell = predict_model(model_tcell, data=best_cancer.drop(columns=["prediction cancer cells"]))
    best_cancer["prediction t-cells"] = pred_tcell["prediction_label"]

    if '[Fab\'CD38-MORF1] (nM)' not in columns:
        st.write(best_cancer[['[Fab\'CD3-MORF2] (nM)','[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)',
                            "prediction cancer cells", "prediction t-cells"]].sort_values(by="prediction t-cells"))
    else:
        st.write(best_cancer[['[Fab\'CD3-MORF2] (nM)','[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)','[Fab\'CD38-MORF1] (nM)',
                            "prediction cancer cells", "prediction t-cells"]].sort_values(by="prediction t-cells"))
        
    pred_ratio = predict_model(model_rec, data=test_data)
    pred_ratio = pred_ratio.rename(columns={"prediction_label":"prediction ratio"})
    best_ratio = pred_ratio[pred_ratio["prediction ratio"] == pred_ratio["prediction ratio"].min()]
    pred_cancer = predict_model(model_cancer, data=best_ratio.drop(columns=["prediction ratio"]))
    best_ratio["prediction cancer"] = pred_cancer["prediction_label"]
    pred_tcell = predict_model(model_tcell, data=best_ratio.drop(columns=["prediction ratio", "prediction cancer"]))
    best_ratio["prediction t-cells"] = pred_tcell["prediction_label"]

    if '[Fab\'CD38-MORF1] (nM)' not in columns:
        st.write(best_ratio[['[Fab\'CD3-MORF2] (nM)','[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)',
                            "prediction cancer", "prediction t-cells", "prediction ratio"]])
    else:
        st.write(best_ratio[['[Fab\'CD3-MORF2] (nM)','[Fab\'CD19-MORF1] (nM)', '[Fab\'CD20-MORF1] (nM)','[Fab\'CD38-MORF1] (nM)',
                            "prediction cancer", "prediction t-cells", "prediction ratio"]])

    # st.pyplot(pred_cancer.plot(x="[Fab\'CD3-MORF2] (nM)", y="prediction cancer cells", kind="scatter").figure)

