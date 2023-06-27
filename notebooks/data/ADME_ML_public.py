from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys
from stat import *
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV, KFold, cross_val_score,RepeatedKFold,train_test_split
import os
import argparse
from argparse import *
from rdkit.Chem.MolStandardize import rdMolStandardize

try:
    set
except NameError:
    from sets import Set as set

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 epilog="--------------Example on HowTo Run ADME_ML_public.py codes ---------------------"+
                                "python ADME_ML_public.py input.sdf -p XXXX -d FCFP4_rdMolDes -m LightGBM -w build ")
parser.add_argument("sdFile",type=str, help="import the molecule pairs of interest ")
parser.add_argument("-p","--properties",required=True, default='XXXX',type=str, help="define the experimental property tag " +
                    "The current property tags are 'LOG HLM_CLint (mL/min/kg)', 'LOG MDR1-MDCK ER (B-A/A-B)', 'LOG SOLUBILITY PH 6.8 (ug/mL)'," +
                    "'LOG RLM_CLint (mL/min/kg)', 'LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)', 'LOG PLASMA PROTEIN BINDING (RAT) (% unbound)' ")
parser.add_argument("-d", "--descriptors",type=str, default='FCFP4_rdMolDes', help="define the descriptor sets to use with _ as the delimiter "+
                    "The available descriptors are rdMolDes, FCFP4")  #############
parser.add_argument("-m", "--model",type=str, default='LightGBM',help="specify the ML algorithms to use "+
                    "The available algorithms is LightGBM, RF, SVM, XGBoost, or Lasso ")
parser.add_argument("-w", "--workflow",type=str, help="specify the task to implement "+
                    "The available modes are build, validation, optimization, prediction ")

args=parser.parse_args()

#How to Run Scripts#
##(build)              python ADME_ML_public.py ADME_HLM.sdf -p 'LOG HLM_CLint (mL/min/kg)' -d FCFP4_rdMolDes -m LightGBM -w build 
##(validation)       python ADME_ML_public.py ADME_HLM.sdf -p 'LOG HLM_CLint (mL/min/kg)' -d FCFP4_rdMolDes -m LightGBM -w validation 
##(optimization)  python ADME_ML_public.py ADME_HLM.sdf -p 'LOG HLM_CLint (mL/min/kg)' -d FCFP4_rdMolDes -m LightGBM -w optimization 
##(prediction)      python ADME_ML_public.py ADME_HLM.sdf -d FCFP4_rdMolDes -w prediction

############################################## 
# 1. Define the input and arguments
##############################################
sdf_file = args.sdFile[:-4]
descType = args.descriptors
ADME_tag = args.properties
ADME_model = args.model
workflow = args.workflow
##############################################


##############################################
# 2. Mol standardization 
##############################################
def standardize(mol):

    try:
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 

        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # try to Canonicalize tautomers
        te = rdMolStandardize.TautomerEnumerator() 
        mol_final = te.Canonicalize(uncharged_parent_clean_mol)
    except:
        try:
            mol_final = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            mol_final = mol

    return mol_final


##############################################
# 3. Generate the descriptors and labels 
##############################################
sdFile=Chem.SDMolSupplier("%s.sdf" % sdf_file)

# get the ADME property values & descriptors
act = {}
maccs = {}
fcfp4_bit = {}
rdMD = {}
name_list = []

i=1
for mol in sdFile:
    if mol is not None:
        mol = standardize(mol)
        try:
            molName = mol.GetProp('_Name')
        except:
            try:
                molName = mol.GetProp('Vendor ID')
            except:
                molName = 'Molecule_%s' %i
        name_list.append(molName)
        try:
            activity = mol.GetProp('%s' % ADME_tag)
        except KeyError:
            activity = '0.0000'  
        act[molName] = float(activity)
        
        #rdMD RDKIT Descriptors
        MDlist = []
        try:
            MDlist.append(rdMolDescriptors.CalcTPSA(mol))
            MDlist.append(rdMolDescriptors.CalcFractionCSP3(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumAmideBonds(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
            MDlist.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))            
            MDlist.append(rdMolDescriptors.CalcNumHeteroatoms(mol))
            MDlist.append(rdMolDescriptors.CalcNumRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
            MDlist.append(rdMolDescriptors.CalcHallKierAlpha(mol))
            MDlist.append(rdMolDescriptors.CalcKappa1(mol))
            MDlist.append(rdMolDescriptors.CalcKappa2(mol))
            MDlist.append(rdMolDescriptors.CalcKappa3(mol))
            MDlist.append(rdMolDescriptors.CalcChi0n(mol))
            MDlist.append(rdMolDescriptors.CalcChi0v(mol))
            MDlist.append(rdMolDescriptors.CalcChi1n(mol))
            MDlist.append(rdMolDescriptors.CalcChi1v(mol))
            MDlist.append(rdMolDescriptors.CalcChi2n(mol))
            MDlist.append(rdMolDescriptors.CalcChi2v(mol))
            MDlist.append(rdMolDescriptors.CalcChi3n(mol))
            MDlist.append(rdMolDescriptors.CalcChi3v(mol))
            MDlist.append(rdMolDescriptors.CalcChi4n(mol))
            MDlist.append(rdMolDescriptors.CalcChi4v(mol))
            MDlist.append(rdMolDescriptors.CalcAsphericity(mol))
            MDlist.append(rdMolDescriptors.CalcEccentricity(mol))   
            MDlist.append(rdMolDescriptors.CalcInertialShapeFactor(mol))
            MDlist.append(rdMolDescriptors.CalcExactMolWt(mol))  
            MDlist.append(rdMolDescriptors.CalcPBF(mol))  
            MDlist.append(rdMolDescriptors.CalcPMI1(mol))
            MDlist.append(rdMolDescriptors.CalcPMI2(mol))
            MDlist.append(rdMolDescriptors.CalcPMI3(mol))
            MDlist.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
            MDlist.append(rdMolDescriptors.CalcSpherocityIndex(mol))
            MDlist.append(rdMolDescriptors.CalcLabuteASA(mol))
            MDlist.append(rdMolDescriptors.CalcNPR1(mol))
            MDlist.append(rdMolDescriptors.CalcNPR2(mol))
            for d in rdMolDescriptors.PEOE_VSA_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.SMR_VSA_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.SlogP_VSA_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.MQNs_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.CalcCrippenDescriptors(mol):
                MDlist.append(d)
            for d in rdMolDescriptors.CalcAUTOCORR2D(mol):  
                MDlist.append(d)
        except:
            print ("The RDdescritpor calculation failed!")

        rdMD[molName] = MDlist

        #Morgan (Circular) Fingerprints (FCFP4) BitVector
        try:
            fcfp4_bit_fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,useFeatures=True,nBits=1024)
            fcfp4_bit[molName] = fcfp4_bit_fp.ToBitString()
        except:
            fcfp4_bit[molName] = ""
            print ("The FCFP4 calculation failed!")                
            
    i = i+1


####################
#Merge descriptors#
####################
dlist = descType.split("_")
combinedheader = []
dtable = {}
fcfp4Test = 1
rdMDTest = 1

#Take the common set of keys among all the descriptors blocks
fcfp4Set = set(fcfp4_bit.keys())
rdMDSet = set(rdMD.keys())
actSet = set(act.keys())

for key in name_list:
    name = key
    if act[key] != "":
        tmpTable = []
        activity = act[key]    

        if "FCFP4" in dlist:
            fcfp4D = fcfp4_bit[key]
            z = fcfp4D.replace('0','0,')
            o = z.replace('1','1,')
            f = o[:-1]   
            fcfp4D = f.split(",") 
            k = 1
            for i in fcfp4D:
                tmpTable.append(i)
                if fcfp4Test:
                    varname = "fcfp4_%d" % k
                    combinedheader.append(varname)
                    k+=1
            fcfp4Test = 0
    
        if "rdMolDes" in dlist:
            rdMD_des = rdMD[key]
            k = 1
            for i in rdMD_des:
                tmpTable.append(str(i))
                if rdMDTest:
                    varname = "rdMD_%d" % k
                    combinedheader.append(varname)
                    k+=1
            rdMDTest = 0   
            
        tmpTable.append(activity)
        dtable[key] = tmpTable
combinedheader.append("activity")
    
#Save out the descriptor file
rawData = open("rawData.csv","w")
for h in combinedheader[:-1]:
    rawData.write("%s," % h)
rawData.write("%s\n" % combinedheader[-1])
for cmpd in dtable.keys():
    comboD = dtable[cmpd]
    rawData.write("%s," % cmpd)
    for d in comboD[:-1]:
        rawData.write("%s," % d)
    rawData.write("%s\n" % comboD[-1])
rawData.close()

##############################################
# 4. Build the ML models
##############################################
n_jobs_model = -1
n_jobs_cv =4

############################################
# set up Hyperparameter search space
############################################
# set up Random Forest parameters

param_base_RF ={'n_estimators': 500, 'oob_score':True,'n_jobs':n_jobs_model}
param_search_RF = {'n_estimators': [100, 250, 500, 750, 1000],
                 'max_features':['sqrt',0.33,0.67, None], 
                 'max_depth': [15, 25, 40, None]} # 5*4*4 =80

# set up SVM parameters
param_base_SVM ={'gamma':'scale'}
param_search_SVM = {'C':[0.1, 1, 5, 10, 20, 50], 
                  'epsilon':[1e-2, 1e-1, 0.3,0.5],
                  'gamma':['scale','auto']} # 6*5 = 30

# set up XGBoost parameters (5+15+3+25+25=73)
param_base_XGB = {'n_estimators': 500, 'subsample':0.8, 'colsample_bytree':0.8, 'n_jobs':n_jobs_model} 
param_search1_XGB = {'n_estimators':[100, 250, 500, 750, 1000]} # 5
param_search2_XGB ={'max_depth':[3,4,5,6,7],'min_child_weight': [1,2,3]} # 5*3 =15
param_search3_XGB = {'gamma':[0, 0.05, 0.1]} # 3
param_search4_XGB = {'subsample':[0.6, 0.7, 0.8, 0.9, 1.0],'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0]} # 5*5 =25
param_search5_XGB = {'reg_alpha':[0, 0.1, 0.2, 0.3, 0.4], 'reg_lambda':[1, 1.1, 1.2, 1.3, 1.4]} # 5*5 =25

param_search_XGB = {'n_estimators':[100, 250, 500, 750, 1000],
                    'max_depth':[3,4,5,6,7],'min_child_weight': [1,2,3],
                    'gamma':[0, 0.05, 0.1],
                    'subsample':[0.6, 0.7, 0.8, 0.9, 1.0],'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha':[0, 0.1, 0.2, 0.3, 0.4], 'reg_lambda':[1, 1.1, 1.2, 1.3, 1.4]} 

# set up LightGBM parameters (5+20+100+16 =141)
param_base_LGB = {'n_estimators': 500, 'subsample':0.8, 'colsample_bytree':0.8,'subsample_freq':1} 
param_search1_LGB = {'n_estimators':[100, 250, 500, 750, 1000]} # 5
param_search2_LGB = {'num_leaves':[15, 31, 45, 60, 75],'min_child_samples':[10, 20, 30, 40]}  # 5* 4 = 20
param_search3_LGB = {'subsample':[0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0], 'subsample_freq': [0,1,3,5]} # 5*5*4 =100
param_search4_LGB = {'reg_alpha':[0, 0.2, 0.5, 0.8], 'reg_lambda':[0, 0.2, 0.5, 0.8]}  # 4*4 = 16

param_search_LGB = {'n_estimators':[100, 250, 500, 750, 1000],
                     'num_leaves':[15, 31, 45, 60, 75],'min_child_samples':[10, 20, 30, 40],
                     'subsample':[0.6, 0.7, 0.8, 0.9, 1.0],'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0],'subsample_freq': [0,1,3,5],
                     'reg_alpha':[0, 0.2, 0.5, 0.8], 'reg_lambda':[0, 0.2, 0.5, 0.8]} 

# set up Lasso parameters (9)
param_base_Lasso = {'alpha': 0.1}
param_search_Lasso = {'alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]}  # 9

# define a Pearson r scoring function
def Pearson_R_score (X,Y):
    pearson_r = pearsonr(X, Y)[0]
    return pearson_r

pearson_r_scorer = make_scorer(Pearson_R_score,greater_is_better=True)

# define a function to do model validation
def model_validation (X_train, Y_train, X_test, Y_test, model):
    # cross-validation
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=128) # 5-fold cross validation for 3 times
    cv_pearson_r = np.average(cross_val_score(model, X_train, Y_train, scoring = pearson_r_scorer, cv=rkf)) ## r2
    # predict for test set
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    r_test = pearsonr(Y_test, Y_pred_test)[0]   
    return cv_pearson_r, r_test

if workflow == 'build':
    os.system("cp rawData.csv trainSet.csv")
    train_df = pd.read_csv("trainSet.csv")
    train_df.dropna(axis=0, how='any', inplace=True)    
    Y_train = train_df["activity"]
    X_train = train_df
    X_train.drop("activity", axis=1, inplace=True)   
    X_train, Y_train = shuffle(X_train, Y_train, random_state=42)       
    
    # train ML model
    if ADME_model == 'RF':
        model = RandomForestRegressor(n_estimators=500,oob_score=True, n_jobs=n_jobs_model)
        model.fit(X_train, Y_train)  
    elif ADME_model == 'SVM':
        scaler = RobustScaler().fit(X_train)
        X_train_new = scaler.transform(X_train)        
        model = SVR(gamma='scale')
        model.fit(X_train_new, Y_train)  
    elif ADME_model == 'XGBoost':
        model = XGBRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8, n_jobs=n_jobs_model)
        model.fit(X_train, Y_train)  
    elif ADME_model == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8,subsample_freq=1, n_jobs=n_jobs_model)
        model.fit(X_train, Y_train)  
    elif ADME_model == 'Lasso':
        scaler = RobustScaler().fit(X_train)
        X_train_new = scaler.transform(X_train)
        model = linear_model.Lasso(alpha=0.1)
        model.fit(X_train_new, Y_train)
    else:
        print("The ML algorithms should be one of RF, SVM, XGBoost, LightGBM, and Lasso")
        
    joblib.dump(model, 'model.rds') 
    os.system("rm rawData.csv")
    
elif workflow == 'validation':
    data_df = pd.read_csv("rawData.csv")
    data_df.dropna(axis=0, how='any', inplace=True) 
    Y_data = data_df["activity"]
    X_data = data_df
    X_data.drop("activity", axis=1, inplace=True) 
    X_data, Y_data = shuffle(X_data, Y_data, random_state=42)     

    # split to training set and hold-out test set
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=84)    
    
    if ADME_model == 'RF':
        model = RandomForestRegressor(n_estimators=500,oob_score=True, n_jobs=n_jobs_model)
        cv_pearson_r, r_test = model_validation (X_train, Y_train, X_test, Y_test, model)
    elif ADME_model == 'SVM':
        model = SVR(gamma='scale')
        scaler = RobustScaler().fit(X_train)
        X_train_new = scaler.transform(X_train) 
        X_test_new = scaler.transform(X_test) 
        cv_pearson_r, r_test = model_validation (X_train_new, Y_train, X_test_new, Y_test, model)
    elif ADME_model == 'XGBoost':
        model = XGBRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8, n_jobs=n_jobs_model)
        cv_pearson_r, r_test = model_validation (X_train, Y_train, X_test, Y_test, model)
    elif ADME_model == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8,subsample_freq=1, n_jobs=n_jobs_model)
        cv_pearson_r, r_test = model_validation (X_train, Y_train, X_test, Y_test, model)
    elif ADME_model == 'Lasso':
        model = linear_model.Lasso(alpha=0.1)
        scaler = RobustScaler().fit(X_train)
        X_train_new = scaler.transform(X_train)
        X_test_new = scaler.transform(X_test)
        cv_pearson_r, r_test = model_validation(X_train_new, Y_train, X_test_new, Y_test, model)
    else:
        print("The ML algorithms should be one of RF, SVM, XGBoost, LightGBM, and Lasso")

    validation_result = pd.DataFrame(index=[ADME_model],columns=['Pearson_r_CV', 'Pearson_r_test'])
    validation_result['Pearson_r_CV'] = [cv_pearson_r]
    validation_result['Pearson_r_test'] = [r_test]
    validation_result.to_csv('%s_%s_default_validation_result.csv'%(sdf_file[5:],ADME_model),index=True, sep=',')
    os.system("rm rawData.csv")
    
elif workflow == 'optimization':
    data_df = pd.read_csv("rawData.csv")
    data_df.dropna(axis=0, how='any', inplace=True) 
    Y_data = data_df["activity"]
    X_data = data_df
    X_data.drop("activity", axis=1, inplace=True) 
    X_data, Y_data = shuffle(X_data, Y_data, random_state=42)     
    # split to training set and hold-out test set
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=84)  
    
    # Hypterparamter tuning fro ML models
    if ADME_model == 'RF':
        model = RandomForestRegressor(n_estimators=500,oob_score=True, n_jobs=n_jobs_model)
        gsearch = GridSearchCV(estimator=model,param_grid = param_search_RF, 
                               scoring='r2',n_jobs=n_jobs_cv, cv=5)        
        gsearch.fit(X_train, Y_train)
        f = open('Hyperparameter_for_%s_RF_result.txt'%sdf_file[5:],'w')
        f.write("Hyperparameter tuning for RandomForest:\n")
        f.write("The best_params are %s \n" % gsearch.best_params_)
        f.write("the best_score of R2 is %s \n" % gsearch.best_score_)
        f.write("\n")
        f.close()
        model.set_params(**gsearch.best_params_)     
        # validation of the optimized model
        cv_pearson_r, r_test = model_validation (X_train, Y_train, X_test, Y_test, model)
                
    elif ADME_model == 'SVM':
        model = SVR(gamma='scale')
        scaler = RobustScaler().fit(X_train)
        X_train_new = scaler.transform(X_train) 
        X_test_new = scaler.transform(X_test) 
        gsearch = GridSearchCV(estimator=model,param_grid = param_search_SVM, 
                               scoring='r2',n_jobs=n_jobs_cv, cv=5)
        gsearch.fit(X_train_new,Y_train)
        f = open('Hyperparameter_for_%s_SVM_result.txt'%sdf_file[5:],'w')
        f.write("Hyperparameter tuning for SVM:\n")
        f.write("The best_params are %s \n" % gsearch.best_params_)
        f.write("The best_score of R2 is %s \n" % gsearch.best_score_) 
        f.write("\n")
        f.close()        
        model.set_params(**gsearch.best_params_)    
        # validation of the optimized model
        cv_pearson_r, r_test = model_validation (X_train_new, Y_train, X_test_new, Y_test, model)
        
    elif ADME_model == 'XGBoost':
        model = XGBRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8, n_jobs=n_jobs_model)
        # Sequential Grid search with 5-fold cross-validation
        gsearch1 = GridSearchCV(estimator=model,param_grid = param_search1_XGB, scoring='r2',n_jobs=n_jobs_cv,cv=5)
        gsearch1.fit(X_train, Y_train)
        f = open('Hyperparameter_for_%s_XGBoost_result.txt'%sdf_file[5:],'w')
        f.write("Hyperparameter tuning for XGBoost:\n")
        f.write("The 1st round of tuning n_estimatros\n")
        f.write("The best_params are %s \n" % gsearch1.best_params_)
        f.write("The best_score is %s \n" % gsearch1.best_score_)  
        f.write("\n")
        model.set_params(**gsearch1.best_params_)          
        
        gsearch2 = GridSearchCV(estimator=model,param_grid = param_search2_XGB, scoring='r2',n_jobs=n_jobs_cv,cv=5)
        gsearch2.fit(X_train, Y_train)            
        f.write("The 2nd round of tuning max_depth & min_child_weight\n")
        f.write("The best_params are %s \n" % gsearch2.best_params_)
        f.write("The best_score is %s \n" % gsearch2.best_score_)  
        f.write("\n")
        model.set_params(**gsearch2.best_params_)  
        
        gsearch3 = GridSearchCV(estimator=model,param_grid = param_search3_XGB, scoring='r2',n_jobs=n_jobs_cv,cv=5)
        gsearch3.fit(X_train, Y_train)    
        f.write("The 3rd round of tuning gamma\n")
        f.write("The best_params are %s \n" % gsearch3.best_params_)
        f.write("The best_score is %s \n" % gsearch3.best_score_)  
        f.write("\n")
        model.set_params(**gsearch3.best_params_)   
        
        gsearch4= GridSearchCV(estimator=model,param_grid = param_search4_XGB, scoring='r2',n_jobs=n_jobs_cv,cv=5)
        gsearch4.fit(X_train, Y_train)
        f.write("The 4th round of tuning sampling parameters\n")
        f.write("The best_params are %s \n" % gsearch4.best_params_)
        f.write("The best_score is %s \n" % gsearch4.best_score_)  
        f.write("\n")
        model.set_params(**gsearch4.best_params_)     
        
        gsearch5 = GridSearchCV(estimator=model,param_grid = param_search5_XGB, scoring='r2',n_jobs=n_jobs_cv,cv=5)
        gsearch5.fit(X_train, Y_train)    
        f.write("The 5th round of tuning regularization parameters\n")
        f.write("The best_params are %s \n" % gsearch5.best_params_)
        f.write("The best_score is %s \n" % gsearch5.best_score_)
        f.write("\n")
        model.set_params(**gsearch5.best_params_)    
        
        f.write("The optimizied model parameters are %s \n" % model.get_params())
        f.close()
        
        # validation of the optimized model
        cv_pearson_r, r_test = model_validation (X_train, Y_train, X_test, Y_test, model)
        
    elif ADME_model == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8,subsample_freq=1, n_jobs=n_jobs_model)
        # Sequential Grid search with 5-fold cross-validation
        gsearch1 = GridSearchCV(estimator=model,param_grid = param_search1_LGB, scoring='r2',n_jobs=n_jobs_cv,cv=5)
        gsearch1.fit(X_train, Y_train)
        f = open('Hyperparameter_for_%s_LightGBM_result.txt'%sdf_file[5:],'w')
        f.write("Hyperparameter tuning for LightGBM:\n")
        f.write("The 1st round of tuning n_estimatros\n")
        f.write("The best_params are %s \n" % gsearch1.best_params_)
        f.write("The best_score is %s \n" % gsearch1.best_score_)  
        f.write("\n")
        model.set_params(**gsearch1.best_params_)      
        
        gsearch2 = GridSearchCV(estimator=model,param_grid = param_search2_LGB, scoring='r2',cv=5, n_jobs=4)
        gsearch2.fit(X_train, Y_train)    
        f.write("The 2nd round of tuning min_child_samples & num_leaves\n")
        f.write("The best_params are %s \n" % gsearch2.best_params_)
        f.write("The best_score is %s \n" % gsearch2.best_score_)  
        f.write("\n")
        model.set_params(**gsearch2.best_params_)         
        
        gsearch3 = GridSearchCV(estimator=model,param_grid = param_search3_LGB, scoring='r2',cv=5, n_jobs=4)
        gsearch3.fit(X_train, Y_train)            
        f.write("The 3rd of tuning sampling parameters\n")
        f.write("The best_params are %s \n" % gsearch3.best_params_)
        f.write("The best_score is %s \n" % gsearch3.best_score_)   
        f.write("\n")
        model.set_params(**gsearch3.best_params_)   
    
        gsearch4 = GridSearchCV(estimator=model,param_grid = param_search4_LGB, scoring='r2',cv=5, n_jobs=4)
        gsearch4.fit(X_train, Y_train)
        f.write("The 4th round of tuning regularization parameters\n")
        f.write("The best_params are %s \n" % gsearch4.best_params_)
        f.write("The best_score is %s \n" % gsearch4.best_score_)  
        f.write("\n")
        model.set_params(**gsearch4.best_params_)        
        
        f.write("The optimizied model parameters are %s \n" % model.get_params()) 
        f.close()  
        # validation of the optimized model    
        cv_pearson_r, r_test = model_validation (X_train, Y_train, X_test, Y_test, model)
        
    elif ADME_model == 'Lasso':
        model = linear_model.Lasso(alpha=0.1)
        scaler = RobustScaler().fit(X_train)
        X_train_new = scaler.transform(X_train)
        X_test_new = scaler.transform(X_test)
        gsearch = GridSearchCV(estimator=model, param_grid=param_search_Lasso,
                               scoring='r2', n_jobs=n_jobs_cv, cv=5)
        gsearch.fit(X_train_new, Y_train)
        f = open('Hyperparameter_for_Lasso_result.txt', 'w')
        f.write("Hyperparameter tuning for Lasso:\n")
        f.write("The best_params are %s \n" % gsearch.best_params_)
        f.write("The best_score of R2 is %s \n" % gsearch.best_score_)
        f.write("\n")
        f.close()
        model.set_params(**gsearch.best_params_)
        # validation of the optimized model
        cv_pearson_r, r_test = model_validation(X_train_new, Y_train, X_test_new, Y_test, model)    
    else:
        print("The ML algorithms should be one of RF, SVM, XGBoost, LightGBM, and Lasso")  
        
    validation_result = pd.DataFrame(index=[ADME_model],columns=['Pearson_r_CV', 'Pearson_r_test'])
    validation_result['Pearson_r_CV'] = [cv_pearson_r]
    validation_result['Pearson_r_test'] = [r_test]
    validation_result.to_csv('%s_%s_Optimized_validation_result.csv'%(sdf_file[5:],ADME_model),index=True, sep=',')     
    os.system("rm rawData.csv")
    
elif workflow == 'prediction':
    os.system("cp rawData.csv testSet.csv")
    test_df = pd.read_csv("rawData.csv")   
    test_df.dropna(axis=0, how='any', inplace=True)
    X_test = test_df
    X_test.drop("activity", axis=1, inplace=True)    
    
    ## loaded pre-trained ADME models
    try:
        loaded_model = joblib.load('model.rds')
    except:
        print('There are no pre-trained model.rds ')
     
    Y_pred_log = loaded_model.predict(X_test) 
    Y_pred = pow(10,Y_pred_log)
    
    sdFileOut = Chem.SDWriter("%s_cADME.sdf" % sdf_file)
    i=0
    for mol in sdFile:
        if mol is not None:
            try:
                molName = mol.GetProp('Name')
            except:
                molName = mol.GetProp('_Name')
            mol.SetProp("cADME", str(Y_pred[i]))
            sdFileOut.write(mol)
        i=i+1
    sdFileOut.close()  
    os.system("rm rawData.csv")
    os.system("rm testSet.csv")

else:
    print("The workflow mush be one of build, validation, optimization, prediction")
            
            
            
    
    


    
