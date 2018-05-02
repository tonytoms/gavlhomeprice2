print ("import")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import re
import operator
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb



print ('reading input files')
#directory="C:\\Users\\Youssef\\Documents\\Data_Science\\Kaggle\\House\\"
#directory="..\input\\"
tr=pd.read_csv("../input/train.csv",sep=",")
te=pd.read_csv("../input/test.csv",sep=",")
#MSSubclass code and descriptions:
#df_subclass=pd.read_csv(directory+"\inter\subclass.csv",sep=",")

col1=[20,    30,    40,    45,    50,    60,    70,    75,    80,    85,    90,    120,150,160,180,190]    
col2=["1-STORY 1946 & NEWER ALL STYLES","1-STORY 1945 & OLDER","1-STORY W/FINISHED ATTIC ALL AGES","1-1/2 STORY - UNFINISHED ALL AGES","1-1/2 STORY FINISHED ALL AGES","2-STORY 1946 & NEWER","2-STORY 1945 & OLDER","2-1/2 STORY ALL AGES","SPLIT OR MULTI-LEVEL","SPLIT FOYER","DUPLEX - ALL STYLES AND AGES","1-STORY PUD (Planned Unit Development) - 1946 & NEWER","1-1/2 STORY PUD - ALL AGES","2-STORY PUD - 1946 & NEWER","PUD - MULTILEVEL - INCL SPLIT LEV/FOYER","2 FAMILY CONVERSION - ALL STYLES AND AGES"]
df_subclass=pd.DataFrame(np.array([col1,col2]).T,columns=["MSSubClass","MSSub_Descr"])

print ("log(Price) and submission Ids")
price=np.log(tr['SalePrice'])
tr=tr.drop(['SalePrice','Id'],axis=1,inplace=False)
Id=te['Id']
te=te.drop(['Id'],axis=1,inplace=False)

print ('concatenate train and test sets')
dftot=pd.concat([tr,te],axis=0)
dftot.reset_index(inplace=True,drop=True)


#To be able to submit, the following step are implemented
#--1    Categorical and non-ordinal features
#--2    Categorical ordinal features
#--3    Specific features adaptations
#--4    Droping garbage
#--5    new/adapted features
#--6    Fill empty values
#--7    Scalling
#--8    XGBoost hyper parameters tuning
#--9    XGBoost prediction
#--10    Submission


#------------------------------------------------------------
print ("1    Catergorical non-ordinal features")
#Getting dummies for Categorical and non-ordinal features
Categ_columns=["MSZoning","Street","Alley","LandContour","LotConfig","LandSlope","BldgType","RoofStyle","RoofMatl","MasVnrType","Foundation","Heating","Electrical","CentralAir","GarageType","PavedDrive","Fence","SaleType","SaleCondition","MiscFeature"]
for col in Categ_columns:
    dftot=pd.concat([dftot,pd.get_dummies(dftot[col],prefix=col)],axis=1)
    dftot=dftot.drop([col],axis=1,inplace=False)    

#------------------------------------------------------------
print ("2    Categorical ordinal features")
#Encoding Categorical ordinal features while keeping the order consistant
#        allowing the estimator to make more efficient splits
#            example: Poor=1, Typical=3, Excellent=5
#        Since the qualitative words are various, this step is repeated many times

#"LotShape"
col1=["Reg","IR1","IR2","IR3"]
col2=[0,1,2,3]
qual=pd.DataFrame(np.array([col1,col2]).T,columns=["LotShape","LotShape_num"])
dftot=pd.merge(dftot,qual,how="left",on="LotShape")
del dftot["LotShape"],col1,col2

#"GarageFinish"
dftot.loc[dftot["GarageFinish"]=="Fin","GarageFinish"]=3
dftot.loc[dftot["GarageFinish"]=="RFn","GarageFinish"]=2
dftot.loc[dftot["GarageFinish"]=="Unf","GarageFinish"]=1


#"ExterQual"
col1=["TA","Gd","Fa","Po","Ex"]
col2=[3,4,2,1,5]
qual=pd.DataFrame(np.array([col1,col2]).T,columns=["ExterQual","ExterQual_num"])
dftot=pd.merge(dftot,qual,how="left",on="ExterQual")
del dftot["ExterQual"],col1,col2

#"ExterCond"
#qual is reused multiple times, but its column names are changed
qual.columns=["ExterCond","ExterCond_num"]
dftot=pd.merge(dftot,qual,how="left",on="ExterCond")
del dftot["ExterCond"]

#"BsmtQual"
qual.columns=["BsmtQual","BsmtQual_num"]
dftot=pd.merge(dftot,qual,how="left",on="BsmtQual")
del dftot["BsmtQual"]

#"BsmtCond"
qual.columns=["BsmtCond","BsmtCond_num"]
dftot=pd.merge(dftot,qual,how="left",on="BsmtCond")
del dftot["BsmtCond"]

#"BsmtExposure"
col1=["Gd","Av","No","Mn"]
col2=[4,3,1,2]
qual1=pd.DataFrame(np.array([col1,col2]).T,columns=["BsmtExposure","BsmtExposure_num"])
dftot=pd.merge(dftot,qual1,how="left",on="BsmtExposure")
del dftot["BsmtExposure"],col1,col2,qual1

#"BsmtFinType1"
col1=["GLQ","ALQ","BLQ","Rec","LwQ","Unf"]
col2=[6,5,4,3,2,1]
qual1=pd.DataFrame(np.array([col1,col2]).T,columns=["BsmtFinType1","BsmtFinType1_num"])
dftot=pd.merge(dftot,qual1,how="left",on="BsmtFinType1")
del dftot["BsmtFinType1"],col1,col2

"BsmtFinType2"
qual1.columns=["BsmtFinType2","BsmtFinType2_num"]
dftot=pd.merge(dftot,qual1,how="left",on="BsmtFinType2")
del dftot["BsmtFinType2"]

"HeatingQC"
qual.columns=["HeatingQC","HeatingQC_num"]
dftot=pd.merge(dftot,qual,how="left",on="HeatingQC")
del dftot["HeatingQC"]

"KitchenQual"
qual.columns=["KitchenQual","KitchenQual_num"]
dftot=pd.merge(dftot,qual,how="left",on="KitchenQual")
del dftot["KitchenQual"]

#"Functional"
col2=["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal"]
col3=[0,1,2,3,4,5,6,7]
qual_2=pd.DataFrame(np.array([col2,col3]).T,columns=["Functional","Functional_num"])
dftot=pd.merge(dftot,qual_2,how="left",on="Functional")
del col2,col3,dftot["Functional"],qual_2

#"FireplaceQu"
qual.columns=["FireplaceQu","FireplaceQu_num"]
dftot=pd.merge(dftot,qual,how="left",on="FireplaceQu")
del dftot["FireplaceQu"]


#"GarageQual"
qual.columns=["GarageQual","GarageQual_num"]
dftot=pd.merge(dftot,qual,how="left",on="GarageQual")
del dftot["GarageQual"]


#"GarageCond"
qual.columns=["GarageCond","GarageCond_num"]
dftot=pd.merge(dftot,qual,how="left",on="GarageCond")
del dftot["GarageCond"]


#"PoolQC"
qual.columns=["PoolQC","PoolQC_num"]
dftot=pd.merge(dftot,qual,how="left",on="PoolQC")
del dftot["PoolQC"]


#------------------------------------------------------------
print ("3    Specific features adaptations")

#"HouseStyle"
#Creating a handmade float corresponding to the number of floors
#with unfinished half floor corresponding to 0.25 floors
col1=["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf"]
col2=[1,1.5,1.25,2,2.5,2.25]
qual=pd.DataFrame(np.array([col1,col2]).T,columns=["HouseStyle","HouseStyle_num"])
dftot=pd.merge(dftot,qual,how="left",on="HouseStyle")

def hs_splitfoyer(x):
    if x=="SFoyer":
        return 1
    else:
        return 0
        
def hs_splitlevel(x):
    if x=="SLvl":
        return 1
    else:
        return 0
            
dftot["hs_splitfoyer"]=dftot["HouseStyle"].apply(hs_splitfoyer)
dftot["hs_splitlevel"]=dftot["HouseStyle"].apply(hs_splitlevel)
del dftot["HouseStyle"],col1,col2



#"Utilities"
# Creating a feature for each utility: Sewer, Sewage and Gas.
#Everyone has electricity so it is useless here.
def util_sewr(x):
    if x=="AllPub":
        return 1
    else:
        return 0

def util_sewg(x):
    if x=="AllPub" or x=="NoSewr":
        return 1
    else:
        return 0
        
def util_gas(x):
    if x=="AllPub" or x=="NoSewr" or x=="NoSeWa" :
        return 1
    else:
        return 0
            
dftot["util_sewr"]=dftot["Utilities"].apply(util_sewr)
dftot["util_sewg"]=dftot["Utilities"].apply(util_sewg)
dftot["util_gas"]=dftot["Utilities"].apply(util_gas)
del dftot["Utilities"]


#"Neighborhood"
#Replacing each neighbourhoud by the logarithm or their mean price
#This feature, if introduced that way, will generate some overfitting but for now it's ok
tempo=pd.concat([tr,np.exp(price)],axis=1)
tab_neigh=pd.DataFrame(tempo.groupby(["Neighborhood"])["SalePrice"].mean()).reset_index()
del tempo
tab_neigh.columns=["Neighborhood","Neigh_meanP"]
dftot=pd.merge(dftot,tab_neigh,how="left",on="Neighborhood")
del tab_neigh,dftot["Neighborhood"]

dftot["Neigh_meanP_log"]=np.log(dftot["Neigh_meanP"])




#"Condition1" & "Condition2" 
#Creating a feature for each condition, because they are independant. 
condit=["Norm","Feedr","PosN","Artery","RRAe","RRNn","RRAn","PosA","RRNe"]
dftot["test"]=dftot["Condition1"]+dftot["Condition2"]

def conditions(condit,x):
    s=[]
    for cond in condit:
        if re.search(cond,x):
            s.append(1)
        else:
            s.append(0)
    return s
    
dftot["test"]=dftot["test"].apply(lambda x: conditions(condit,x))
temp=pd.DataFrame(np.array(list(dftot["test"])),columns=condit)
dftot=pd.concat([dftot,temp],axis=1)
del dftot["test"],temp,dftot["Condition1"],dftot["Condition2"]



#------------------------------------------------------------
print ("4    Droping")
#For now, I don't do anything with Exterior1st and Exterior2nd
dftot=dftot.drop(['Exterior1st','Exterior2nd'],axis=1,inplace=False)

#------------------------------------------------------------
print ("5    new/adapted features")

#"MoSold" & "YrSold"
#Replacing these two column by one integer keeping the order of the events:
#for example 12/15 is replaced by 15*100+12=1512
dftot["date_sale"]=(dftot["YrSold"].astype(str).apply(lambda x: x[2:4])).astype(int)*100+dftot["MoSold"]
del dftot["MoSold"],dftot["YrSold"]


#'MSSubClass'
#Extracting the features: Before 1945 and After 1946
#using the MSSubClass codes descriptions that have been imported initially

dftot=pd.merge(dftot,df_subclass,how="left",on="MSSubClass")
def subclass_b45(x):
    if re.search("AGES",str(x)):
        return 1
    elif re.search("1945 & OLDER",str(x)):
        return 1
    else:
        return 0
def subclass_a46(x):
    if re.search("AGES",str(x)):
        return 1
    elif re.search("1946 & NEWER",str(x)):
        return 1
    else:
        return 0

dftot["MSSubClass_b45"]=dftot["MSSub_Descr"].apply(subclass_b45)
dftot["MSSubClass_a46"]=dftot["MSSub_Descr"].apply(subclass_a46)
dftot=dftot.drop(["MSSubClass","MSSub_Descr"],axis=1)

#"Area derivative"
#Basement unfinished ratio
dftot["Bsmt_unfin"]=dftot["BsmtUnfSF"]/dftot["TotalBsmtSF"]

#Basement finished average quality
# The average is based on the area of each finished floor type
dftot["Bsmt_mean_qual"]=(dftot["BsmtFinSF1"]*dftot["BsmtFinType1_num"].astype(float)+dftot["BsmtFinSF2"]*dftot["BsmtFinType2_num"].astype(float))/(dftot["BsmtFinSF1"]+dftot["BsmtFinSF2"])

#Total porch and Total porch+deck
dftot["Bsmt_tot_porch"]=dftot["OpenPorchSF"]+dftot["EnclosedPorch"]+dftot["3SsnPorch"]+dftot["ScreenPorch"]
dftot["Bsmt_tot_porchdeck"]=dftot["OpenPorchSF"]+dftot["EnclosedPorch"]+dftot["3SsnPorch"]+dftot["ScreenPorch"]+dftot["WoodDeckSF"]

#------------------------------------------------------------
print ('6    Fill empty values')
#Filling all NA values with 0. Only one house is partially completed. So this
#assumption doesn't have too much effect
dftot = dftot.fillna(value=0)



#------------------------------------------------------------
print ("7    Conversion to numeric & Scalling")
#With the operations above, all the converted categorical ordinal features are
#still objects. Let's turn them into numbers
dftot["HouseStyle_num"]=dftot["HouseStyle_num"].astype(float)
mergenum=["ExterQual_num","ExterCond_num","BsmtQual_num","BsmtCond_num",
        "BsmtExposure_num","BsmtFinType1_num","BsmtFinType2_num","HeatingQC_num",
        "KitchenQual_num","Functional_num","FireplaceQu_num","GarageQual_num",
        "GarageCond_num","PoolQC_num","LotShape_num"]
        
dftot[mergenum]=dftot[mergenum].astype(int)

#YearBuilt" & "YearRemodAdd"
#min and max scaling
mini=dftot["YearBuilt"].min()
dftot["YearBuilt"]=(dftot["YearBuilt"]-mini)/(dftot["YearBuilt"].max()-mini)
mini=dftot["YearRemodAdd"].min()
dftot["YearRemodAdd"]=(dftot["YearRemodAdd"]-mini)/(dftot["YearRemodAdd"].max()-mini)
del mini

#for all other columns, we apply a maximum scaling
for col in dftot.columns:
    mm=dftot[col].max()
    if mm>1:
        dftot[col]=dftot[col].astype(float)/float(mm)


#------------------------------------------------------------
#---------------END OF TRANSFORTMATION-----------------------
#------------------------------------------------------------


#------------------------------------------------------------
print ("8    XGBoost hyper parameters tuning")

#Splitting back train and test set
feat=np.array(dftot[0:len(price)].copy())
test=np.array(dftot[len(price)::].copy())

##xxx=XGBRegressor(base_score=1, colsample_bylevel=0.9, colsample_bytree=0.65, 
#        gamma=0.003,learning_rate=0.1, max_delta_step=0, max_depth=6,
#       min_child_weight=3.5, missing=None, n_estimators=1000, nthread=-1,
#       objective='reg:linear', reg_alpha=0, reg_lambda=0,
#       scale_pos_weight=1, seed=0, silent=True, subsample=1)


dtrain_tot=xgb.DMatrix(feat,label=price)


param = {"booster" : "gbtree", 'silent':1, 'objective':'reg:linear' }
param['eval_metric'] = "rmse"

#The tuning occurs in three steps
# First tuning: eta and number of boosting trees
param['eta']=0.03
n_round=1000

# Second tuning: Tree parameters. Increasing model complexity until it doesn't improve performance
param['max_depth']=6
param['min_child_weight'] = 3.5
param['gamma'] = 0.003

# Third tuning: Limiting overfitting by:
#Introducing randomness
param['colsample_bylevel'] =1
param['colsample_bytree'] = 1
#Regularization
param['reg_lambda'] = 1
param['reg_alpha'] = 0


# Using XGB.CV allows us to better assess performance by considering the random error of choosing a train set and a test set.
grid=xgb.cv(param, dtrain_tot, num_boost_round=n_round, nfold=10,
           stratified=False, folds=None, metrics=('rmse'), 
           obj=None, feval=None, maximize=False, 
           early_stopping_rounds=100, fpreproc=None, 
           as_pandas=True, verbose_eval=50, show_stdv=True, 
           seed=12, callbacks=None)

#plotting train and test error
plt.plot(range(50,len(grid)),grid["test-rmse-mean"][50::],range(50,len(grid)),grid["train-rmse-mean"][50::],'r')
plt.show()

#------------------------------------------------------------
print ("9    XGBoost prediction assessment")
#After a bit of tuning, I found the following parameters:

param2 = {"booster" : "gbtree", 'silent':1, 'objective':'reg:linear' }
param2['eval_metric'] = "rmse"
param2['eta']=0.03
n_round2=1000
param2['max_depth']=6
param2['min_child_weight'] = 3.5
param2['gamma'] = 0.003
param2['colsample_bylevel'] =1
param2['colsample_bytree'] = 1
param2['reg_lambda'] = 1
param2['reg_alpha'] = 0


#final training with all the training data available
bst=xgb.train(param2,dtrain_tot, n_round2,
           obj=None, feval=None, maximize=False, 
           early_stopping_rounds=None, verbose_eval=50,
            callbacks=None)


#plotting relative features importance
#plotting only 20 most important features        
importance = sorted((bst.get_fscore()).items(), key=operator.itemgetter(1))
df_importance = pd.DataFrame(importance, columns=['feature', 'fscore'])
for j in range(0,len(importance)):
   df_importance.loc[j,'feature']=dftot.columns[int((importance[j][0])[1:len(importance[j][0])])]
df_importance['fscore'] = df_importance['fscore'] / df_importance['fscore'].sum()
plt.figure()
df_importance.tail(20).plot(kind='barh',x='feature',y='fscore')            



#------------------------------------------------------------    
print ("10    Submission")
#Getting back the price
output1=np.exp(bst.predict(xgb.DMatrix(test)))
submission=pd.DataFrame()
submission['Id']=Id
submission['SalePrice']=pd.Series(output1)
submission.to_csv("output.csv",index=False,sep=",")

#Leaderboard results: 0.12686
