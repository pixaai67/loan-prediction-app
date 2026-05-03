import pickle
import pandas as pd
import gradio as gr
with open("best_model.pkl", "rb") as f:
    best_model=pickle.load(f)
#fun
def predict_loan(
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area
):
    #dta inp
    input_data=pd.DataFrame(
        [{
        "Gender":Gender,
        "Married":Married,
        "Dependents":str(Dependents),
        "Education":Education,
        "Self_Employed":Self_Employed,
        "ApplicantIncome":ApplicantIncome,
        "CoapplicantIncome":CoapplicantIncome,
        "LoanAmount":LoanAmount,
        "Loan_Amount_Term":Loan_Amount_Term,
        "Credit_History":Credit_History,
        "Property_Area":Property_Area
    }]
        )
    #fea
    input_data['TotalIncome']=input_data['ApplicantIncome']+input_data['CoapplicantIncome']
    input_data['LoanToIncomeRatio']=input_data['LoanAmount']/(input_data['TotalIncome']+1)
    input_data['EMI']=input_data['LoanAmount']/(input_data['Loan_Amount_Term']/12)
    #prdct
    prediction=best_model.predict(input_data)[0]
    proba=best_model.predict_proba(input_data)[0]
    if prediction=="Y":
        return f"Approved:{proba[1]*100:.1f}%"
    else:
        return f"Rejcted:{proba[0]*100:.1f}%" 
with gr.Blocks(title="Loan Approval Prediction System",theme=gr.themes.Soft()) as app:
    gr.Markdown("Loan Approval Prediction System\nEnter applicant details below")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Personal Information")
            Gender=gr.Dropdown(["Male","Female"],label="Gender",value="Male")
            Married=gr.Dropdown(["Yes","No"],label="Married",value="No")
            Dependents=gr.Dropdown(["0","1","2","3"],label="Dependents",value="0")
            Education=gr.Dropdown(["Graduate","Not Graduate"],label="Education",value="Graduate")
            Self_Employed=gr.Dropdown(["Yes","No"],label="Self Employed",value="No")
            Property_Area=gr.Dropdown(["Urban","Rural","Semiurban"],label="Property Area",value="Urban")
        with gr.Column(scale=1):
            gr.Markdown("Financial Information")
            ApplicantIncome=gr.Number(label="Applicant Income",value=5000,step=1000)
            CoapplicantIncome=gr.Number(label="Coapplicant Income",value=0,step=1000)
            LoanAmount=gr.Number(label="Loan Amount",value=5000,step=10)
            Loan_Amount_Term=gr.Number(label="Loan Term (mnts)",value=120,step=12)
            Credit_History=gr.Dropdown([1,0],label="Credit History(1=Gd,0=Bd)", value=1)
    #btn
    predict_btn=gr.Button("Predict Loan Status",variant="primary",size="lg")
    output_result=gr.Textbox(label="Prediction Result",lines=3,interactive=False)
    predict_btn.click(
        fn=predict_loan,
        inputs=[
            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area
        ],
        outputs=output_result
    )
    gr.Markdown("Prediction based on a LR ML model trained on loan dataset")
#lnch
app.launch(share=True)