import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
#ld dtast
df=pd.read_csv("train_u6lujuX_CVtuZ9i (1) (2).csv")
print(df.head())
print("shape:-",df.shape)
#presng
df_cln=df.copy()
df_cln["Dependents"]=df_cln["Dependents"].fillna("0")
df_cln["Dependents"]=df_cln["Dependents"].replace("3+","3")
df_cln["TotalIncome"]=df_cln["ApplicantIncome"]+df_cln["CoapplicantIncome"]
df_cln["LoanToIncomeRatio"]=df_cln["LoanAmount"]/(df_cln["TotalIncome"]+1)
df_cln["EMI"]=df_cln["LoanAmount"]/(df_cln["Loan_Amount_Term"]/12)
df_cln["Loan_Status_Encoded"]=df_cln["Loan_Status"].map({"Y": 1,"N": 0})
#fra & trgt
X=df_cln.drop(["Loan_ID", "Loan_Status", "Loan_Status_Encoded"],axis=1)
y=df_cln["Loan_Status"]
num_cls=["ApplicantIncome",
           "CoapplicantIncome",
           "LoanAmount",
           "Loan_Amount_Term",
           "TotalIncome",
           "LoanToIncomeRatio", 
           "EMI"
           ]
cat_cls=["Gender",
         "Married",
         "Dependents",
         "Education",
         "Self_Employed",
         "Property_Area"
         ]
bin_cls=["Credit_History"]
#ppln
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
cat_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore",drop="first"))
])
bin_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler",StandardScaler())
])
preprocessor=ColumnTransformer([
    ("num",num_pipeline,num_cls),
    ("cat",cat_pipeline,cat_cls),
    ("bin",bin_pipeline,bin_cls)
])
#trn tst 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#mdl ppln
mdl_pipeline=Pipeline([
    ("preprocessor",preprocessor),
    ("model",LogisticRegression(max_iter=1000,random_state=42))
])
#cv
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_scor=cross_val_score(mdl_pipeline,X_train,y_train,cv=cv,scoring="accuracy")
print("cv mean-:",cv_scor.mean())
print("cv std-:",cv_scor.std())
#hypr tung
param_grid={"model__C":[0.001,0.01,0.1,1,10,100]}
grid_search=GridSearchCV(
    estimator=mdl_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("bst params   :", grid_search.best_params_)
print("bst cv score :", grid_search.best_score_)
#bst mdl
best_model=grid_search.best_estimator_
#eva
trn_sc=best_model.score(X_train,y_train)
tst_sc=best_model.score(X_test,y_test)
y_pred=best_model.predict(X_test)
y_proba=best_model.predict_proba(X_test)[:,1]
prec=precision_score(y_test,y_pred,pos_label="Y")
rec=recall_score(y_test,y_pred,pos_label="Y")
f1=f1_score(y_test,y_pred,pos_label="Y")
roc_auc=roc_auc_score(y_test,y_proba)
print("trn acc:-",trn_sc)
print("tst acc-:",tst_sc)
print("presn-:",prec)
print("rcll-:",rec)
print("f1-:",f1)
print("roc auc:",roc_auc)
#sv mdl
with open("best_model.pkl","wb") as f:
    pickle.dump(best_model,f)
print("saved best_model.pkl")