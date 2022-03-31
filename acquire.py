#%%
from webbrowser import get
from env import host, user, password
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from tabulate import tabulate

#%%
def get_titanic_data():
    filename = 'titanic_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'titanic_db'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'

    query = "SELECT* FROM passengers"


    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
# %%

# %%
# The returned data frame should include the actual name 
# of the species in addition to the species_ids.
#%%
def get_iris_data():
    filename = 'iris_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'iris_db'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'

    query = '''
    SELECT *
    FROM measurements
    JOIN species USING (species_id)

    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
# %%

# %%
# join all 4 tables together, so that the resulting dataframe 
# contains all the contract, payment, and internet service options.
#%%
def get_telco_data():
    filename = 'telco_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'telco_churn'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'

    query = '''
    SELECT*
    FROM customers
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id)

    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
#%%
def get_attendance_data():
    filename = 'attendance_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'tidy_data'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'


    query = '''
    SELECT *
    FROM attendance
    

    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
#%%
def get_coffee_data():
    filename = 'coffee_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'tidy_data'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'


    query = '''
    SELECT *
    FROM coffee_levels
    

    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
# %%
def get_cake_data():
    filename = 'cake_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'tidy_data'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'


    query = '''
    SELECT *
    FROM cake_recipes
    

    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df

#%%

def get_zillow_data():
    filename = 'zillow_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'zillow'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'


    query = '''
    SELECT parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, poolcnt, pooltypeid2, hashottuborspa , 
    fireplacecnt, garagecarcnt , numberofstories, airconditioningtypeid, heatingorsystemtypeid, yearbuilt, lotsizesquarefeet, latitude, longitude, regionidcounty, 
    regionidzip, fips, taxvaluedollarcnt, transactiondate, propertylandusetypeid
    FROM `properties_2017`
    JOIN `predictions_2017` USING (parcelid)
    WHERE `propertylandusetypeid` = "261" and transactiondate LIKE '2017%%'
    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
#%% 
def get_zillow_all():
    filename = 'zillow_data.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
    
    database = 'zillow'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'


    query = '''
    SELECT *
    FROM `properties_2017`
    JOIN `predictions_2017` USING (parcelid)
    JOIN story  
    WHERE `propertylandusetypeid` = "261" and transactiondate LIKE '2017%%'
    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, url)
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df
