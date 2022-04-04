#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, RocCurveDisplay
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, RobustScaler, QuantileTransformer


import seaborn as sns
from copy import deepcopy

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
#%%
classifications_list = [
    LogisticRegression(max_iter=500), 
    KNeighborsClassifier(n_neighbors=7), DecisionTreeClassifier(max_depth=3), RandomForestClassifier(n_estimators=1000, criterion='entropy',max_features='sqrt'),
    ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(learning_rate=0.01, loss='exponential', max_depth=3, n_estimators=1000, subsample=0.5), 
    MLPClassifier(activation='logistic', alpha=0.0001, learning_rate='constant', solver='adam'),
    RidgeClassifier(alpha=0.4)
]

#%%
classifierList = [
    LogisticRegression(max_iter=500), 
    KNeighborsClassifier(n_neighbors=7), DecisionTreeClassifier(max_depth=3), RandomForestClassifier(n_estimators=1000, criterion='entropy',max_features='sqrt'),
    ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(learning_rate=0.01, loss='exponential', max_depth=3, n_estimators=1000, subsample=0.5), 
    MLPClassifier(activation='logistic', alpha=0.0001, learning_rate='constant', solver='adam'),
    RidgeClassifier(alpha=0.4)
]
#%%

#%%
def model_magic(X_train, y_train, X_validate, y_validate, X_test, y_test, classifierList):
    """
    Description
    ----
    This function tests out all models listed in classifierList.
    
    If the model isn't valid, the function prints out the invalid
    name of the model along with it's error.
    The accuracy score, confusion matric, positive precision, recall 
    and f-scores, and negative prevision, recall and f-scores.
    
    Data is then appended into a dictionary with columns.
    
    
    Parameters
    ----
    The X_train split for the dataframe.
        
    The X_test split for the dataframe.

    The X_validate split for dataframe.

    The y_validate splot for dataframe.

    The y_train split for the dataframe.
        
    The y_test split for the dataframe.
        
    classifierList (list of models):
        The list of models chosen.
    
    Returns
    ----
    dic (dataframe):
        A dataframe of dic.
        
    """
    # Cretate Dictionary for model stats train, validate, test 
    dic = {'ModelName': [], 'AccuracyScore': [], 'AccuracyScoreVAL': [],
           'CorrectPredictionsCount': [], 'CorrectPredictionsCountVAL': [], 'Total': [], 'TotalVAL': [], 
           'PosPrecScore': [], 'PosPrecScoreVAL':[], 'PosRecScore': [], 'PosRecScoreVAL': [] ,'PosFScore': [],
           'PosFScoreVAL': [],'NegPrecScore': [], 'NegPrecScoreVAL': [] ,'NegRecScore': [], 'NegRecScoreVAL': [],
           'NegFScore': [], 'NegFScoreVAL': [], 'TNPercentage': [], 'TNPercentageVAL': [],'TPPercentage': [], 
           'TPPercentageVAL': [],'FNPercentage': [], 'FNPercentageVAL': [], 'FPPercentage': [], 'FPPercentageVAL': []}
    
    # Deepcopy the classifierList
    models = deepcopy(classifierList)
    
    # Test each models in the list to verify 
    for i in range(len(classifierList)):
        try:
            model = classifierList[i]
            model.fit(X_train, y_train)
        except Exception as e:
            print("==============================================================")
            print(f"I wasn't able to score with the model: {classifications_list[i]}")
            print(f"This was the error I've received from my master:\n\n{e}.")
            print("\nI didn't let it faze me though, for now I've skipped this model.")
            print("==============================================================\n")
            models.remove(classifierList[i]) # Remove invalid models from list
    
    # Loop through all models
    for classifier in range(len(models)):
        # removes any 
        modelName = re.sub(r"\([^()]*\)", '', str(models[classifier]))
        # Performance
        model = models[classifier]
        model.fit(X_train, y_train)          
        pred = model.predict(X_test)
        pred1 = model.predict(X_validate)
        # Results
        acc_score = accuracy_score(y_test, pred)
        acc_score1 = accuracy_score(y_validate, pred1) 
        noOfCorrect = accuracy_score(y_test, pred, normalize = False)
        noOfCorrect1 = accuracy_score(y_validate, pred1, normalize = False) 
        total = noOfCorrect/acc_score
        total1 = noOfCorrect1/acc_score1
        Confusing = confusion_matrix(y_test, pred)
        madConfusing1 = confusion_matrix(y_validate, pred1)
        # calculations 
        dpps = Confusing[1][1] / (Confusing[1][1] + Confusing[0][1]) # pos prec score
        dpps1 = madConfusing1[1][1] / (madConfusing1[1][1] + madConfusing1[0][1])
        dprs = Confusing[1][1] / (Confusing[1][1] + Confusing[1][0]) # pos rec score
        dprs1 = madConfusing1[1][1] / (madConfusing1[1][1] + madConfusing1[1][0])
        dpfs = 2 * (dpps * dprs) / (dpps + dprs) # pos f1 score
        dpfs1 = 2 * (dpps1 * dprs1) / (dpps1 + dprs1) # pos f1 score
        dnps = Confusing[0][0] / (Confusing[0][0] + Confusing[1][0]) # neg prec score
        dnps1 = madConfusing1[0][0] / (madConfusing1[0][0] + madConfusing1[1][0])
        dnrs = Confusing[0][0] / (Confusing[0][0] + Confusing[0][1]) # neg rec score
        dnrs1 = madConfusing1[0][0] / (madConfusing1[0][0] + madConfusing1[0][1])
        dnfs = 2 * (dnps * dnrs) / (dnps + dnrs) # neg f1 score
        dnfs1 = 2 * (dnps1 * dnrs1) / (dnps1 + dnrs1) 
               

        # Save Calulations and append to dictionary 
        dic['ModelName'].append(modelName)
        dic['AccuracyScore'].append(acc_score)
        dic['AccuracyScoreVAL'].append(acc_score1)
        dic['CorrectPredictionsCount'].append(noOfCorrect)
        dic['CorrectPredictionsCountVAL'].append(noOfCorrect1)
        dic['Total'].append(total)
        dic['TotalVAL'].append(total1)
        dic['PosPrecScore'].append(dpps)
        dic['PosPrecScoreVAL'].append(dpps1)
        dic['PosRecScore'].append(dprs)
        dic['PosRecScoreVAL'].append(dprs1)
        dic['PosFScore'].append(dpfs)
        dic['PosFScoreVAL'].append(dpfs1)
        dic['NegPrecScore'].append(dnps)
        dic['NegPrecScoreVAL'].append(dnps1)
        dic['NegRecScore'].append(dnrs)
        dic['NegRecScoreVAL'].append(dnrs1)
        dic['NegFScore'].append(dnfs)
        dic['NegFScoreVAL'].append(dnfs1)
        dic['TNPercentage'].append(Confusing[0][0]/total*100)
        dic['TNPercentageVAL'].append(madConfusing1[0][0]/total*100)
        dic['TPPercentage'].append(Confusing[1][1]/total*100)
        dic['TPPercentageVAL'].append(madConfusing1[1][1]/total*100)
        dic['FNPercentage'].append(Confusing[1][0]/total*100)
        dic['FNPercentageVAL'].append(madConfusing1[1][0]/total*100)
        dic['FPPercentage'].append(Confusing[0][1]/total*100)
        dic['FPPercentageVAL'].append(madConfusing1[0][1]/total*100)
        
    return pd.DataFrame.from_dict(dic)


#%%
def corrstatsgraphs(df):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap with target label, and outputs a distribution plot.
    
    Parameters
    ----
    df(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    useful stats, correlation, and subplots
    
    """
    # Description
    print("Descriptive Stats:")
    display(df.describe().T)
    
    # Heatmap with min -1 to max 1 to all variables
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    corr = df.corr()
    f, ax = plt.subplots(figsize=(22, 17))
    plt.title("Heatmap", fontsize = 'x-large')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 21, as_cmap=True)
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr 
    plt.figure(figsize=(10, 8)) 
    plt.title("Heatmap", fontsize = 'x-large')
    sns.heatmap(df.corr()[['taxvaluedollarcnt']].sort_values(by='taxvaluedollarcnt', 
    ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG'
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr
    plt.figure(figsize=(16,10))
    df.corr()['taxvaluedollarcnt'].sort_values(ascending=False).plot(kind='bar', figsize=(20,5), cmap='BrBG'
    )

    
#%%
#%%
def magic2(edf):
    """
    Description
    ----
    Splits a single given dataframe using 
    'train_test_split'
    
    
    Parameters
    ----
    df (dataframe):
        The dataframe to use for modeling.
    
    test_size (float):
        The test_size that you want to give for 
        train_test_split.
        The default test_size is set to 0.2 for 
        test_validate and test
        The default test_size is set to 0.3 for
        train and validate
    
    Returns
    ----
    save (dataframe):
        The dataframe after running the splits in
        `model_magic`
        
    """
    
    # Split
    train_validate, test = train_test_split(edf, test_size=0.2, random_state=42, stratify=edf['churn'])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=42, stratify=train_validate['churn'])
    # get dummmys 
    dummy_train = pd.get_dummies(train[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_validate = pd.get_dummies(validate[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_test = pd.get_dummies(test[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    #merge dummies with orginal dataframe
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    #drop columns with corresponding dummies
    train = train.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    validate = validate.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    test = test.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    # assign x, y trian, validate, test 
    X_train = train.drop(columns=['customer_id','churn_Yes'])
    y_train = train.churn_Yes

    X_validate = validate.drop(columns=['customer_id','churn_Yes'])
    y_validate = validate.churn_Yes

    X_test = test.drop(columns=['customer_id','churn_Yes'])
    y_test = test.churn_Yes

    save = model_magic(X_train, y_train, X_validate, y_validate, X_test, y_test, classifierList)

    
    
    return save
#%%
def prep_zillow(pdf):
    """
    Converts to_numeric, drops rows with NaN values, splits data using sklean.train_test_split, hot encodes with pd.get_dummies(),
    concats dataframe with dummies, drops original dummies, and assigns X, y variables to train, validate, and test.

    Returns:
    X_train, y_train, X_validate, y_validate, X_test, y_test
    
    """
    # convert total_charges to float64 error will be NaN
    pdf['total_charges'] = pd.to_numeric(pdf['total_charges'], errors='coerce')
    
    # Drop NaN rows since they are 0 in total_charges and reset_index to 0 
    pdf = pdf.dropna(axis=0)
    pdf.reset_index(drop=True)

    # Split data into train, validate, test 
    train_validate, test = train_test_split(pdf, test_size=0.2, random_state=42, stratify=pdf['churn'])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=42, stratify=train_validate['churn'])
    # hot encoding using pd.get_dummies for non-numbrical catagorical data for train, validate, test
    dummy_train = pd.get_dummies(train[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_validate = pd.get_dummies(validate[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    dummy_test = pd.get_dummies(test[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn']], drop_first=[True])
    # concat pdf dataframe with dummies 
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    # drop original columns 
    train = train.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    validate = validate.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    test = test.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                            'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 
                            'streaming_movies', 'paperless_billing', 'churn'])
    
    return train, validate, test
#%%
# CHI-SQUARED TEST FUNCTION to idenify non-corralations between target

def chi2_test(train, target, cat_features):

    '''
    Loop function to create pd.crosstab of target and features that correlates & not correlates with target.
    If it does not correlates to append to new list call Remove.

    '''
    # list of non-corralations features to be removed from X_train, X_validate, X_test
    Remove=[]
    print('The Chi2_test result are : \n')
    for feature in cat_features: 
        CrossResult=pd.crosstab(index = train[target], columns=train[feature])
        # p-value is index [1]
        pval = chi2_contingency(CrossResult)[1]
        # print(Result)
        # If the ChiSquare P-Value is <0.05, that means we reject H0
        if (pval < 0.05):
            print(feature, 'correlates with', target, '| P-Value:', pval)
        else:
            print(feature, 'does not correlates with', target, '| P-Value:', pval) 
            # append to remove list  
            Remove.append(feature)     
    print("\n\n")
    # return list 
    return(Remove)
#%%%
## PEARSONR CORRELATION TEST FUNCTION
def correlation_test(df, target, num_features):
    '''
    Given two subgroups from a dataset, conducts a correlation test for linear relationship between df and target.
    Utilizes the method provided in the Codeup curriculum for conducting correlation test using
    scipy and pandas. 
    '''
    # list of non-corralations features to be removed from X_train, X_validate, X_test
    Remove=[]

    print('The T-Test result are :\n')
    for feature in num_features:
        feature_num = df.groupby(target)[feature].apply(list)
        # p-value is index [1]
        pvalue = ttest_ind(*feature_num)[1]
        # If the T-Test P-Value is <0.05
        if (pvalue < 0.05):
            print(feature, 'correlates with', target , '| P-Value:', pvalue)
        else:
            print(feature, ' NOT correlates with', target , '| P-Value:', pvalue)
            Remove.append(feature)
    print("\n\n")
    # returns list 
    return(Remove)
#%%%
def X_y(train, validate, test):

    # drop target data for X_train and assign target data for y_train
    X_train = train.drop(columns=['taxvalue','county'])
    y_train = train.taxvalue
    # drop target data for X_validate and assign target data for y_validate
    X_validate = validate.drop(columns=['taxvalue'])
    y_validate = validate.taxvalue
    # drop target data for X_test and assign target data for y_test 
    X_test = test.drop(columns=['taxvalue'])
    y_test = test.taxvalue
    
        
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#%%
def wrangle_grades(df):

    """
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    """
    # Acquire data from csv file.
    grades = pd.read_csv("student_grades.csv")
    # Replace white space values with NaN values.
    grades = grades.replace(r"^\s*$", np.nan, regex=True)
    # Drop all rows with NaN values.
    df = grades.dropna()
    # Convert all columns to int64 data types.
    df = df.astype("int")
    return df
#%%
def X_full_y_full(df):
    # used to split train to figure out the best imputer method 
    # drop target data for X_train and assign target data for y_train
    X_full = df.drop(columns=['id','propertylandusetypeid'], axis=1)
    y_full = df.drop(columns=['id','propertylandusetypeid'])
    
    
    return X_full, y_full

#%%
def wrangle_zillow(df_prep):
    """
    Converts NaN values in bathroom/bedrooms/pool,pool w/hot tube/spa, has spa/hot tub to value 0, replace
    air & heating with correct values and impute with NaN to most frequent value, drop top four outliers in taxvalue,
    drops rows with NaN values, convert fips code to county and rename column to county, rename all columns for readablity,
    splits data using sklean.train_test_split. 
    

    Returns:
    train, validat, test
    
    """
    # Since there is not a good way to calulate missing values regarding bedrooms & bathrooms and there are
    # only 137 values total I will be converting them to np.NaN using the replace function.
    bed_bath_0_columns = ['bedroomcnt', 'bathroomcnt']
    df_prep[bed_bath_0_columns] = df_prep[bed_bath_0_columns].replace(0, np.NaN)
    df_prep = df_prep.reset_index(drop=True)
    # The NaN values in fireplaces, num_pool, pool_w_spa_or_hottub, has_hottub_or_spa, and garage_cars are mostly 0 will be   
    # using the replace function and replacing the NaN values. 
    fire_garge_pool_0_columns = ['poolcnt','pooltypeid2','hashottuborspa', 'fireplacecnt','garagecarcnt']
    df_prep[fire_garge_pool_0_columns] = df_prep[fire_garge_pool_0_columns].replace(np.NaN, 0)
    df_prep = df_prep.reset_index(drop=True)

    # replace value 5 with 0 since 5 = no a/c replace value 13 with 1 since 13 = a/c yes
    df_prep.airconditioningtypeid.replace(5, 0, inplace=True) # No AC   
    df_prep.airconditioningtypeid.replace(13, 1, inplace=True) # Central AC

    # replace value 13 with 0 since 13 = no heating and replace NaN with most frequent value
    df_prep.heatingorsystemtypeid.replace(13, 0, inplace=True) # no heat
    df_prep.heatingorsystemtypeid.replace(np.NaN, 2, inplace=True) # replace nan with 2 most frequent value 

    # replace NaN value with moth frequent value 1
    df_prep.airconditioningtypeid.replace(np.NaN, 1, inplace=True) # Central AC    

    # drop columns'parcelid', 'propertylandusetypeid', 'transactiondate' dont coorilate with the target
    df_prep = df_prep.drop(columns=['parcelid', 'propertylandusetypeid', 'transactiondate'])

    # to make it easier to read and improve visualization I will be renameing the columns using the rename function
    df_prep = df_prep.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'sqft_living',
    'poolcnt':'num_pools', 'pooltypeid2':'pool_w_spa_or_hottub','hashottuborspa':'has_hottub_or_spa' ,'fireplacecnt':'fireplaces', 
    'garagecarcnt':'garage_cars', 'airconditioningtypeid':'ac_type','heatingorsystemtypeid':'heating_type' ,'yearbuilt':'year_built', 'lotsizesquarefeet':'sqft_lot', 
    'latitude':'lat', 'longitude':'long', 'regionidcounty':'countyid', 'regionidzip':'zip', 'fips':'fips','taxvaluedollarcnt': 'taxvalue'}
    )
    # drop all rows with NaN values
    df_prep = df_prep.dropna(axis=0)
    df_prep = df_prep.reset_index(drop=True)

    # drop the top 4 upper outliers in taxvalue
    df_prep.drop([2874,8881,11695,6787], axis=0, inplace=True)

    # convert fips to county 
    df_prep['fips'] = df_prep['fips'].replace(6111.0, 'Ventura, CA')
    df_prep['fips'] = df_prep['fips'].replace(6059.0, 'Orange, CA')
    df_prep['fips'] = df_prep['fips'].replace(6037.0, 'Los Angeles, CA')

    # drop countyid & long/lat columns
    df_prep.drop(columns=['countyid'], inplace=True)
    df_prep.drop(columns=['long'], inplace=True)
    df_prep.drop(columns=['lat'], inplace=True)

    # rename fips to county
    df_prep = df_prep.rename(columns={'fips':'county'})

    #change categorical features to categorical dtype
    df_prep.zip = df_prep.zip.astype('category')
    df_prep.county = df_prep.county.astype('category')
    df_prep.ac_type = df_prep.ac_type.astype('category')
    df_prep.heating_type = df_prep.heating_type.astype('category')
    df_prep.has_hottub_or_spa = df_prep.has_hottub_or_spa.astype('category')
    df_prep.pool_w_spa_or_hottub = df_prep.pool_w_spa_or_hottub.astype('category')
    df_prep.numberofstories = df_prep.numberofstories.astype('category')
    df_prep.fireplaces = df_prep.fireplaces.astype('category')
    df_prep.garage_cars = df_prep.garage_cars.astype('category')
    df_prep.year_built = df_prep.year_built.astype('category')
    df_prep.num_pools = df_prep.num_pools.astype('category')
    
    df_prep.drop(columns=['ac_type','heating_type','has_hottub_or_spa','pool_w_spa_or_hottub','numberofstories','fireplaces','garage_cars','year_built','num_pools'], inplace=True)

    train_validate, test = train_test_split(df_prep, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)
    return train, validate, test
#%%
def corrstatsgraphs3(df):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap with target label, and outputs a distribution plot.
    
    Parameters
    ----
    df(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    useful stats, correlation, and subplots
    
    """
       
    # Heatmap with min -1 to max 1 to all variables
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    corr = df.corr()
    plt.subplots(figsize=(22, 17))
    plt.title("Heatmap", fontsize = 'x-large')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 21, as_cmap=True)
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap, vmin=-1, vmax=1
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr 
    plt.figure(figsize=(10, 8)) 
    plt.title("Heatmap", fontsize = 'x-large')
    sns.heatmap(df.corr()[['taxvalue']].sort_values(by='taxvalue', 
    ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG'
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr
    plt.figure(figsize=(16,10))
    df.corr()['taxvalue'].sort_values(ascending=False).plot(kind='bar', figsize=(20,5), cmap='BrBG'
    )
        
    sns.jointplot(x="bedrooms", y="taxvalue", data=df,  kind='reg', height=5, line_kws={'color': 'red'}
    )
    sns.jointplot(x="bedrooms", y="taxvalue", data=df,  kind='kde', line_kws={'color': 'red'}
    )
    sns.jointplot(x="bathrooms", y="taxvalue", data=df, ratio=5, kind='reg', height=5, line_kws={'color': 'red'}
    )
    sns.jointplot(x="bathrooms", y="taxvalue", data=df,  kind='kde', line_kws={'color': 'red'}
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    r, p = stats.pearsonr(df.sqft_living, df.taxvalue)
    sns.scatterplot(data=df, x=df.sqft_living, y=df.taxvalue,ci=None, ax=ax)
    ax.set(
    xlabel='Sqft Living',
    ylabel='tax value',
    title='Sqft Living vs. tax value',
    )
    text = f'r = {r:.4f}, p = {p:.4f}'
    ax.text(25, 120, text, va='top', ha='right'
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    r, p = stats.pearsonr(df.sqft_lot, df.taxvalue)
    sns.scatterplot(data=df, x=df.sqft_lot, y=df.taxvalue,ci=None)
    ax.set(
    xlabel='sqft lot',
    ylabel='tax value',
    title='Sqft Living vs. tax value',
    )
    text = f'r = {r:.4f}, p = {p:.4f}'
    ax.text(25, 120, text, va='top', ha='right')
    plt.show()
#%%
def zillow_data_prep(train, validate, test):

        
    dummy_train = pd.get_dummies(train[['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built','county', 'zip']], drop_first=[True])
    dummy_validate = pd.get_dummies(validate[['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built','county', 'zip']], drop_first=[True])
    dummy_test = pd.get_dummies(test[['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built','county', 'zip']], drop_first=[True])
    #merge dummies with orginal dataframe
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    #drop columns with corresponding dummies
    train = train.drop(columns=['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built','county', 'zip'])
    validate = validate.drop(columns=['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built','county', 'zip'])
    test = test.drop(columns=['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built','county', 'zip'])


#%%
def bs_X_y(bs_train, bs_validate, bs_test):
    '''
    
    '''
    # drop target data for X_train and assign target data for y_train
    X_train = bs_train.drop(columns=['taxvalue', 'county'])
    y_train = bs_train.taxvalue
    # drop target data for X_validate and assign target data for y_validate
    X_validate = bs_validate.drop(columns=['taxvalue', 'county'])
    y_validate = bs_validate.taxvalue
    # drop target data for X_test and assign target data for y_test 
    X_test = bs_test.drop(columns=['taxvalue','county'])
    y_test = bs_test.taxvalue
    
        
    return X_train, y_train, X_validate, y_validate, X_test, y_test
    
        
#%%
def zillow_data_prep_1(train_1, validate_1, test_1):
    # get_dummies for categorical features
    dummy_train = pd.get_dummies(train_1[['bedrooms', 'bathrooms']], drop_first=[True])
    dummy_validate = pd.get_dummies(validate_1[['bedrooms', 'bathrooms']], drop_first=[True])
    dummy_test = pd.get_dummies(test_1[['bedrooms', 'bathrooms']], drop_first=[True])
    # merge dummies with orginal dataframe
    train_1 = pd.concat([train_1, dummy_train], axis=1)
    validate_1 = pd.concat([validate_1, dummy_validate], axis=1)
    test_1 = pd.concat([test_1, dummy_test], axis=1)
    # drop columns with corresponding dummies
    train_1 = train_1.drop(columns=['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built', 'county', 'zip'])
    validate_1 = validate_1.drop(columns=['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built', 'county', 'zip'])
    test_1 = test_1.drop(columns=['bedrooms', 'bathrooms', 'num_pools','pool_w_spa_or_hottub', 'has_hottub_or_spa', 'fireplaces','garage_cars', 'numberofstories', 'heating_type',
    'ac_type','year_built', 'county', 'zip'])
    return train_1, validate_1, test_1

#%%
def scale_data(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    '''
    columns_to_scale = ['sqft_living','sqft_lot']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = RobustScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
#%%
def data(train, validate, test): 
    '''
    changes numeric data to categorical data
    
    '''
    
    train.bedrooms = train.bedrooms.astype('category')
    train.bathrooms = train.bathrooms.astype('category')
    train.zip = train.zip.astype('category')
    train.county = train.county.astype('category')
    
    validate.bedrooms = validate.bedrooms.astype('category')
    validate.bathrooms = validate.bathrooms.astype('category')
    validate.zip = validate.zip.astype('category')
    validate.county = validate.county.astype('category')
    
    test.bedrooms = test.bedrooms.astype('category')
    test.bathrooms = test.bathrooms.astype('category')
    test.zip = test.zip.astype('category')
    test.county = test.county.astype('category')
    return train, validate, test

#%%
def get_dummies(train, validate, test):
    dummy_train = pd.get_dummies(train[['bedrooms', 'bathrooms']], drop_first=[True])
    dummy_validate = pd.get_dummies(validate[['bedrooms', 'bathrooms']], drop_first=[True])
    dummy_test = pd.get_dummies(test[['bedrooms', 'bathrooms']], drop_first=[True])
    # merge dummies with orginal dataframe
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    # drop columns with corresponding dummies
    train = train.drop(columns=['bedrooms', 'bathrooms'])
    validate = validate.drop(columns=['bedrooms', 'bathrooms'])
    test = test.drop(columns=['bedrooms', 'bathrooms'])
    return train, validate, test

#%%
def X_y_split(train, validate, test):
    """
    Description
    ----
    Splits the dataframe into X and y variables.
    
    Parameters
    ----
    df(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    X and y variables
    
    """
       
    X_train = train.drop(columns=['taxvalue', 'county', 'zip'])
    y_train = train.taxvalue
    X_validate = validate.drop(columns=['taxvalue', 'county', 'zip'])
    y_validate = validate.taxvalue
    X_test = test.drop(columns=['taxvalue', 'county', 'zip'])
    y_test = test.taxvalue
    return X_train, y_train, X_validate, y_validate, X_test, y_test
#%%
def corrstatsgraphs4(df):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap with target label, and outputs a distribution plot.
    
    Parameters
    ----
    df(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    useful stats, correlation, and subplots
    
    """
       
    # Heatmap with min -1 to max 1 to all variables
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    corr = df.corr()
    plt.subplots(figsize=(22, 17))
    plt.title("Correlation Heatmap Zillow", fontsize = 'x-large')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 21, as_cmap=True)
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap, vmin=-1, vmax=1
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr 
    plt.figure(figsize=(10, 8)) 
    plt.title("Heatmap", fontsize = 'x-large')
    sns.heatmap(df.corr()[['taxvalue']].sort_values(by='taxvalue', 
    ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG'
    )
#%%
def remove_outliers(df, k, col_list):
    ''' 
    This function remove outliers from a list of columns in a dataframe 
    and returns that dataframe
    '''
    
    # loop through each column
    for col in col_list:
        
        # Get the quantiles
        q1, q3 = df[col].quantile([.25, .75])
        
        # Get the quantile range
        iqr = q3 - q1
        
        # Establish the upper and lower
        upper_bound = q3 + k * iqr  
        lower_bound = q1 - k * iqr   

        # Redefine the DataFrame with removed outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#%%
def corrstatsgraphs5(df):
    """
    Description
    ----
    Outputs the general statistical description of the dataframe,
    outputs the correlation heatmap with target label, and outputs a distribution plot.
    
    Parameters
    ----
    df(DataFrame):
        The dataframe for which information will be displayed.
        
    Returns
    ----
    useful stats, correlation, and subplots
    
    """
       
    # Heatmap with min -1 to max 1 to all variables
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    corr = df.corr()
    plt.subplots(figsize=(22, 17))
    plt.title("Correlation Heatmap Zillow", fontsize = 'x-large')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 21, as_cmap=True)
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap, vmin=-1, vmax=1
    )
    # Correlation Heatmap with min -1 to max 1 in conjuction with pd.corr 
    plt.figure(figsize=(10, 8)) 
    plt.title("Heatmap", fontsize = 'x-large')
    sns.heatmap(df.corr()[['logerror']].sort_values(by='logerror', 
    ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG'
    )

#%%
def wrangle_zillow_all(df_prep):
    """
    Converts NaN values in bathroom/bedrooms/pool,pool w/hot tube/spa, has spa/hot tub to value 0, replace
    air & heating with correct values and impute with NaN to most frequent value, drop top four outliers in taxvalue,
    drops rows with NaN values, convert fips code to county and rename column to county, rename all columns for readablity,
    splits data using sklean.train_test_split. 
    

    Returns:
    train, validat, test
    
    """
    # Since there is not a good way to calulate missing values regarding bedrooms & bathrooms and there are
    # only 137 values total I will be converting them to np.NaN using the replace function.
    bed_bath_0_columns = ['bedroomcnt', 'bathroomcnt']
    df_prep[bed_bath_0_columns] = df_prep[bed_bath_0_columns].replace(0, np.NaN)
    df_prep = df_prep.reset_index(drop=True)
    # The NaN values in fireplaces, num_pool, pool_w_spa_or_hottub, has_hottub_or_spa, and garage_cars are mostly 0 will be   
    # using the replace function and replacing the NaN values. 
    fire_garge_pool_0_columns = ['poolcnt','pooltypeid2','hashottuborspa', 'fireplacecnt','garagecarcnt']
    df_prep[fire_garge_pool_0_columns] = df_prep[fire_garge_pool_0_columns].replace(np.NaN, 0)
    df_prep = df_prep.reset_index(drop=True)

    # replace value 5 with 0 since 5 = no a/c replace value 13 with 1 since 13 = a/c yes
    df_prep.airconditioningtypeid.replace(5, 0, inplace=True) # No AC   
    df_prep.airconditioningtypeid.replace(13, 1, inplace=True) # Central AC

    # replace value 13 with 0 since 13 = no heating and replace NaN with most frequent value
    df_prep.heatingorsystemtypeid.replace(13, 0, inplace=True) # no heat
    df_prep.heatingorsystemtypeid.replace(np.NaN, 2, inplace=True) # replace nan with 2 most frequent value 

    # replace NaN value with moth frequent value 1
    df_prep.airconditioningtypeid.replace(np.NaN, 1, inplace=True) # Central AC    

    # drop columns'parcelid', 'propertylandusetypeid', 'transactiondate' dont coorilate with the target
    df_prep = df_prep.drop(columns=['parcelid', 'propertylandusetypeid', 'transactiondate'])

    # to make it easier to read and improve visualization I will be renameing the columns using the rename function
    df_prep = df_prep.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'sqft_living',
    'poolcnt':'num_pools', 'pooltypeid2':'pool_w_spa_or_hottub','hashottuborspa':'has_hottub_or_spa' ,'fireplacecnt':'fireplaces', 
    'garagecarcnt':'garage_cars', 'airconditioningtypeid':'ac_type','heatingorsystemtypeid':'heating_type' ,'yearbuilt':'year_built', 'lotsizesquarefeet':'sqft_lot', 
    'latitude':'lat', 'longitude':'long', 'regionidcounty':'countyid', 'regionidzip':'zip', 'fips':'fips','taxvaluedollarcnt': 'taxvalue'}
    )
    # drop all rows with NaN values
    df_prep = df_prep.dropna(axis=0)
    df_prep = df_prep.reset_index(drop=True)

    # drop the top 4 upper outliers in taxvalue
    df_prep.drop([2874,8881,11695,6787], axis=0, inplace=True)

    # convert fips to county 
    df_prep['fips'] = df_prep['fips'].replace(6111.0, 'Ventura, CA')
    df_prep['fips'] = df_prep['fips'].replace(6059.0, 'Orange, CA')
    df_prep['fips'] = df_prep['fips'].replace(6037.0, 'Los Angeles, CA')

    # drop countyid & long/lat columns
    df_prep.drop(columns=['countyid'], inplace=True)
    df_prep.drop(columns=['long'], inplace=True)
    df_prep.drop(columns=['lat'], inplace=True)

    # rename fips to county
    df_prep = df_prep.rename(columns={'fips':'county'})

    #change categorical features to categorical dtype
    df_prep.zip = df_prep.zip.astype('category')
    df_prep.county = df_prep.county.astype('category')
    df_prep.ac_type = df_prep.ac_type.astype('category')
    df_prep.heating_type = df_prep.heating_type.astype('category')
    df_prep.has_hottub_or_spa = df_prep.has_hottub_or_spa.astype('category')
    df_prep.pool_w_spa_or_hottub = df_prep.pool_w_spa_or_hottub.astype('category')
    df_prep.numberofstories = df_prep.numberofstories.astype('category')
    df_prep.fireplaces = df_prep.fireplaces.astype('category')
    df_prep.garage_cars = df_prep.garage_cars.astype('category')
    df_prep.year_built = df_prep.year_built.astype('category')
    df_prep.num_pools = df_prep.num_pools.astype('category')
    
    df_prep.drop(columns=['ac_type','heating_type','has_hottub_or_spa','pool_w_spa_or_hottub','numberofstories','fireplaces','garage_cars','year_built','num_pools'], inplace=True)

    train_validate, test = train_test_split(df_prep, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)
    return train, validate, test