""" Helper module for pre-processing the various datasets.
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
from collections import Counter
from scipy import stats
import math
import seaborn as sns
sns.set(style="whitegrid")

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def get_appliation_df(name):
    """ Get train or testing dataset.
    """
    df = pd.read_csv('data_raw/{}'.format(name))

    #convert these fields to float64
    cols = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_ID_PUBLISH', \
        'HOUR_APPR_PROCESS_START','CNT_CHILDREN']

    for c in cols:
        df[c]  = df[c].astype('float64')

    return df

def get_bureau():
    """ Get the bureau dataset.
    """
    df =  pd.read_csv('data_raw/bureau.csv')

    cols = ['DAYS_CREDIT','CREDIT_DAY_OVERDUE','CNT_CREDIT_PROLONG','DAYS_CREDIT_UPDATE']

    for c in cols:
        df[c]  = df[c].astype('float64')

    return df

def get_bureau_balance():
    """ Get the bureau balances dataset.
    """
    return pd.read_csv('data_raw/bureau_balance.csv')

def get_pos_cash_balances():
    """ Get pos cash balances dataset.
    """
    df = pd.read_csv('data_raw/POS_CASH_balance.csv')

    cols = ['MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF']

    for c in cols:
        df[c]  = df[c].astype('float64')

    return df

def get_credit_card_balance():
    """ Get credit card balances dataset dataset.
    """
    df = pd.read_csv('data_raw/credit_card_balance.csv')
    cols = ['MONTHS_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','CNT_DRAWINGS_CURRENT','SK_DPD','SK_DPD_DEF']

    for c in cols:
        df[c]  = df[c].astype('float64')

    return df

def get_previous_application():
    """ Get the previous applications dataset.
    """
    return pd.read_csv('data_raw/previous_application.csv')

def get_combined_dataset():
    """ Get the combined dataset.
    """
    return pd.read_csv('data_merged/combined_dataset.csv')

def get_full_train_norm():
    """ Get the combined dataset.
    """
    return pd.read_csv('data_final/train_norm.csv')

def get_stratified_train():
    """ Get the stratified train dataset.
    """
    return pd.read_csv('data_stratified/train.csv')

def get_stratified_train_norm():
    """ Get the stratified normalized dataset.
    """
    return pd.read_csv('data_stratified/train_norm.csv')

def get_full_train_norm():
    """ Get the full normalized dataset.
    """
    return pd.read_csv('data_full/train_norm.csv')

def get_full_test_norm():
    """ Get the full normalized dataset.
    """
    return pd.read_csv('data_full/test_norm.csv')

def get_stratified_test_norm():
    """ Get the stratified normalized dataset.
    """
    return pd.read_csv('data_stratified/test_norm.csv')

def get_installments_payments():
    """ Get intallments payments dataset.
    """
    df = pd.read_csv('data_raw/installments_payments.csv')
    cols = ['NUM_INSTALMENT_NUMBER']

    for c in cols:
        df[c]  = df[c].astype('float64')

    return df

def get_train_df():
    ''' Get the application training dataset'''
    return get_appliation_df('application_train.csv')
    
def get_test_df():
    ''' Get the application testing dataset'''
    return get_appliation_df('application_test.csv')

def get_application_df():
    """ Get the train and testing dataset
        combined as one dataset.  
        
        The returned dataframe has a column called
        DATASET to filter the training and testing:
            train => the training dataset
            test => the test dataset
    """
    train =  get_appliation_df('application_train.csv')
    train['DATASET'] = 'train'
    test =  get_appliation_df('application_test.csv')
    test['DATASET'] = 'test'

    df = train.append(test,ignore_index=True,sort=False)

    return df

def applyMax(features,df,percent=99):
    """ Applys a max value on given fields based on
        the given percentile

        Params
        ------
        features:   list
                    The fields we want apply a max value.
        
        df:         DataFrame
                    The DataFrame the has the data we want to set a max value too.
        
        percent:    int
                    The percentile rate [ default is 99]
    """
    for feature in features:

        #check if any missing values
        has_na = df[feature].isna().values.any()
        assert not has_na, "{} has nan values.".format(feature)

        max_val = int(np.percentile(df[feature], percent)) 
        df[feature] =  df[feature].apply(lambda x:x if x < max_val else max_val)

class FieldsNames(object):
    """
        Creates a dataframe with dataset name with
        fields and fields descriptions. Import
        from HomeCredit_columns_description.csv file
    """
    def __init__(self):
        
        #import into dataframe
        df =  pd.read_csv('data_raw/HomeCredit_columns_description.csv', encoding = "ISO-8859-1")

        grps = df.groupby('Table')

        #lets create a dictionary for each dataset
        #along with its rows and description
        self.table = dict()
        for name , grp in grps:
            
            #get rows and descriptions as list
            fields = grp['Row'].values
            decsr = grp['Description'].values
            

            temp_dict = dict()
            for f,d in zip(fields,decsr):
                temp_dict[f] = d

            self.table[name]= temp_dict

    def print(self,dataset_name,rows:list):

        msg = "{}) ['{}']: " + "{}\n" + "-" * 40
        count = 1
        for row,descr in self.table[dataset_name].items():
            # print(row)
            if row in rows:
                print(msg.format(count ,row,descr))
                # print("\n")
                count += 1

class ABT(object):
    """
        Creates Activity Base Table (ABT [1] ) from dataframe

        Params
        -------
        df: pandas DataFrame
            The dataset we want to create ABT for.
        
        ref: [1]
         Fundamentals of Machine Learning for Predictive Data Analytics:
         by Kellerher, Namee, & D'Arcy
    """
    def __init__(self,df: pd.DataFrame):
        
        self.field_types = df.dtypes.value_counts()
        
        self.col_count = len(df.columns)

        #excluded fields
        exclude= ['SK_ID_CURR','TARGET','DATASET','SK_ID_BUREAU', 'SK_ID_PREV']
        self.cols =   [c for c in  df.columns.values.tolist() if c not in exclude]

        self.types =  [t.name for t in df[self.cols].dtypes]
    
        int_indx = []
        float_indx = []
        obj_indx = []

        for idx,t in enumerate(self.types):
            if 'int' in t:
                int_indx.append(idx)
            elif 'float' in t:
                float_indx.append(idx)
            else:
                obj_indx.append(idx)

        #dataframe length
        self.length = len(df)

        #list of continouse features
        if len(float_indx) > 0:
            self.float_features = [self.cols[i] for i in float_indx] 
            #get float describe table
            self.float_table = self.float_abt(df)

        if len(int_indx) > 0:
            #list of int features
            self.int_features = [self.cols[i] for i in int_indx ] 

            #get int describe
            self.int_table = self.get_categorical(df[self.int_features])

        if len(obj_indx) > 0:
            #list of categorical features
            self.other_features = [self.cols[i] for i in obj_indx ]

            #get other describe table
            self.other_table = self.get_categorical(df[self.other_features])

    def info(self):
        """ Print field type info.
        """
        print("Column count: {}".format(self.col_count))
        print("-" * 20)

        print("Row count: {}".format(self.length))
        print("-" * 20)

        print("Types:")
        print("-" * 20)
        ids = self.field_types.index
        vals = self.field_types.values
        for i,v in zip(ids,vals):
            print("Type: {} Count: {}".format(i,v))

    def get_categorical(self,df):
        """ Get the categorical ABT table

            Params
            ------
            df: pandas DataFrame
                The dataset we want to create ABT for.
        """

        count_df = len(df)
        na_count = df.isna().sum().values.tolist()
        cols = df.columns.values.tolist()

        count = []
        unique = []
        top = []
        freq = []
        for c in cols:
            cat = df[c].astype('category').describe()
            count.append(cat.loc['count'])
            unique.append(cat.loc['unique'])
            top.append(cat.loc['top'])
            freq.append(cat.loc['freq'])

        #create dict
        order_dict = OrderedDict()
        order_dict['num'] = list(range(1,len(count)+1))
        order_dict['count'] = count
        order_dict['na_count'] = na_count
        order_dict['na_%'] = [0] * len(cols)
        order_dict['unique'] =  unique
        order_dict['top'] =  top
        order_dict['top_count'] =  freq
        
        #create dataframe
        new_df = pd.DataFrame(order_dict,index=cols)
        new_df.index.name = 'column'

        #update count
        new_df['top_%'] = new_df['top_count']/count_df
        new_df['na_%'] =  new_df['na_count']/count_df

        return new_df 

    def float_table_nan_filter(self,threshold,include):
        """ Filters out fields with nan % greater
            than threshold rate.

            Params
            ------
            threshold:  float
                        Only select fields with nan rates less
                        than this rate.

            Returns
            -------
            Float fields table as a DataFrame
        """
        ids = self.float_table[self.float_table['na_%'] < threshold].index.values.tolist()
        ids = ids + include 
        return self.float_table.loc[ids]

    def other_table_nan_filter(self,threshold):
        """ Filters out fields with nan % greater
            than threshold rate.

            Params
            ------
            threshold:  float
                        Only select fields with nan rates less
                        than this rate.

            Returns
            -------
            Float fields table as a DataFrame
        """
        ids = self.other_table[self.other_table['na_%'] < threshold].index
        return self.other_table.loc[ids]
    
    def get_float_table(self,fields):
        """ Get float fields ABT.
        
            Params
            ------
            fields: list or array
                    The fields to include in the ABT

            Returns
            -------
            returns DataFrame
        """
        return self.float_table.loc[fields]
        
    def float_abt(self,df):
        descr = df[self.float_features].describe().T
        descr['num'] = list(range(1,len(descr)+1))
        descr['na_count'] = self.length - descr['count']
        descr['na_%'] =  descr['na_count']/self.length

        #reorder columns
        cols = ['num', 'count','na_count','na_%']
        cols_= [i for i in descr.columns.values.tolist() if i not in cols]
        return descr[cols +  cols_]

class Scaler(object):
    """
        Class re-scaler to reduce skewness and outliers.
        Create a new dataframe and then applies re-scaling transformations.

        Params
        ------
        fields: list
                    The fields we want include in the new dataframe

        df: DataFrame
              Dataset that contains the training and testing combined.

    """
    def __init__(self,features,df):
        
        #first lets make a copy of the dataset
        self.data = df[features].copy()

        #add reference to original dataset
        self.df = df

    def trans_log(self,features:list,fillna=True):
        """ Apply log transformation to 
            given feature.

            Params
            ------
            features:   list
                        The fields we want to apply a log 
                        transformation too.
        """
        
        for feature in features:
            #add 1 to handle zeros.
            self.data[feature] =self.data[feature].apply(lambda x: np.log(x+1))

        #fill any na
        if fillna:
            self.fillNa()   
    
    def make_positive(self,features:list):
        """ Make every value in column absolute.

            Params
            ------
            features:   list
                        The fields we want to make obsolute.
        """
        for feature in features:
            self.data[feature] =self.data[feature].apply(lambda x: abs(x) +.01)

    def days_employed_cat(self):
        """ Make days employed into a category field. 
            Also any positive amounts represents 
            retired applicants and should be grouped
            as the last bin.

            Params
            -----
            bins:   int (Defualt is 365 days or 1 year)
                    The number of days grouping count.    
        """
        bins=730
        self.data['DAYS_EMPLOYED']  = self.data['DAYS_EMPLOYED'].apply(lambda x: int(x/bins))
        
        #We need to make int bin posivite.
        # since days worked are negative find the min int group
        # to use as the max value.
        max_val = (self.data['DAYS_EMPLOYED'].min()) * - 1
        max_val += 1


        def year_cat(x):
            if x <= 5:
                return x
            elif x>5 and x <= 10:
                return 10
            elif x>10 and x <= 15:
                return 15
            else:
                return 20

        #make all positive
        self.data['DAYS_EMPLOYED']  = self.data['DAYS_EMPLOYED'].apply(lambda x: year_cat(abs(x)) if x <= 0 else max_val )

        self.data['DAYS_EMPLOYED'] = self.data['DAYS_EMPLOYED'].astype('float64')


    def remove_outliers(self,outlier_fields:list, step_size=1.5):
        """ Removes outliers from given feature 
            using tukeys method.

            Params
            ------
            outlier_fields:     list
                                The fields we want to remove outliers from.
        """
        self.step_size = step_size

        #we only want to remove outliers from training dataset
        train_df = self.data[self.data['DATASET']=='train']
        outlier_ids = self.get_outlier_ids_tukey(train_df,outlier_fields)

        #create new DataFrame that has outliers removed.
        self.data = self.data.drop(outlier_ids).copy().reset_index(drop=True)
        
        #lets create  descriptive stats
        self.describe_df =  self.data[self.data['DATASET']=='train'].describe().T

    def get_outlier_ids_tukey(self,data,features=[]):
        """ Get the index ids for the outlier row 
            using tukey's method.

             Params
            ------
            data: DataFrame
            
            features: list
                    The fields we want to get apply tukey's method too.
        """
        counter = Counter()
        for feature in features:
            
            #filter out missing values
            df = data[data[feature].notna()][feature]

            # Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(df, 25)

            # Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(df, 75)

            # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            step = (Q3-Q1)*self.step_size

            min_val = Q1 - step
            max_val = Q3 + step
            print("{}: Min {:.4f}, Max {:.4f}".format(feature,min_val ,max_val))
            ids = ~((df >= min_val) & (df <= max_val))
            counter.update(df[ids].index.values.tolist())

        return list(counter.keys())

    def boxcox(self,features:list):
        """ Perform boxcox transformations on given features. 

            Params
            ------
            features:     list
                            The fields we want to perform
                            boxcox transformation on.
        """
        
        for feature in features:
            #first perform box on training dataset
            #get ids of training datasets
            ids = self.data_train[self.data_train[feature].notna()].index.values
            
            #perform boxcox
            vals = self.data_train[self.data_train[feature].notna()][feature].values
            x = stats.boxcox(vals)

            #update training values
            X_update = pd.DataFrame(x[0], columns=[feature], index=ids)
            self.data.update(X_update)

            #now update test set using lmbda from training set
            ids_test = self.data_test[self.data_test[feature].notna()].index.values
            vals_test = self.data_test[self.data_test[feature].notna()][feature].values
            x_test = stats.boxcox(vals_test,lmbda=x[1])

            #update training values
            X_update_test = pd.DataFrame(x_test, columns=[feature], index=ids_test)
            self.data.update(X_update_test)
            print("Boxcox complete for {}".format(feature))

    def fillNa(self):
        """ Fills any missing values with median
            on given features. 
        """

        #get list of all nan features
        nan_list =   self.data.isna().sum()
        idx = nan_list.index
        counts = nan_list.values

        nan_features = []
        for i,v in zip(idx,counts):
            if v > 0 and i not in ['SK_ID_CURR','TARGET','DATASET']:
                nan_features.append(i)
             
        #first create median value table
        self.describe_df =  self.data[self.data['DATASET']=='train'].describe().T

        #fill any missing values
        for feature in nan_features:
            self.data[feature].fillna(self.describe_df.at[feature,'50%'],inplace=True )

        #lets create descriptive stats again after filling in missing values.
        self.describe_df =  self.data[self.data['DATASET']=='train'].describe().T

    def apply(self,feature,fun):
        """ Applys a function to a given feature.

            Params
            ------
            features:   list
                        The fields we want apply a function.
        """
        self.data[feature] = self.data[feature].apply(fun)

    def applyMax(self,features):
        """ Applys a max value on given fields based on
            the 99.99% percentile

            Params
            ------
            features:   list
                        The fields we want apply a max value.
        """
        for feature in features:

            #check if any missing values
            has_na = self.data[feature].isna().values.any()
            # print("has na {}".format(has_na))
            assert not has_na, "{} has nan values.".format(feature)

            max_val = np.percentile(self.data[feature], 99.99)
            self.data[feature] =  self.data[feature].apply(lambda x:x if x < max_val else max_val)

    @property
    def data_train(self):
        ''' Get training dataset '''
        return self.data[self.data['DATASET']=='train'] 

    @property
    def data_test(self):
        ''' Get testing dataset '''
        return self.data[self.data['DATASET']=='test'] 

def plot_hbar(fields,df,w=20,h=7):
    """ Plot histogram for each given field.
    
        Params
        ------
        fields: list
                The list of columns we want to plot.
                
        df:     DataFrame
                Our Dataset.
    
    """
    sns.set_color_codes("muted")

    nrows = int(math.ceil(len(fields) / 4))

    fig, axs = plt.subplots(nrows, 4,figsize=(w,int(nrows * h)))

    for ax, i in zip(axs.flat, fields):
        cnt = df[df[i].notna()].groupby(i).size()
        sns.barplot(x=cnt.values,y=cnt.index ,  ax=ax,  color="b")
        ax.set_ylabel(i)
        ax.set_xlabel("Count")

        #format axes
        # ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks()])

    #hide unused axis
    unused = len(axs.flat) - len(fields)
    if unused > 0:
        for a in  axs.flat[-unused:]:
            a.axis('off')  

    plt.tight_layout()
    plt.show()

def plot_bar(fields,df,w=20,h=7):
    """ Plot histogram for each given field.
    
        Params
        ------
        fields: list
                The list of columns we want to plot.
                
        df:     DataFrame
                Our Dataset.
    
    """
    sns.set_color_codes("muted")

    nrows = int(math.ceil(len(fields) / 4))

    fig, axs = plt.subplots(nrows, 4,figsize=(w,int(nrows * h)))

    for ax, i in zip(axs.flat, fields):
        cnt = df[df[i].notna()].groupby(i).size()
        sns.barplot(y=cnt.values,x=cnt.index ,  ax=ax,  color="b")
        ax.set_xlabel(i)
        ax.set_ylabel("Count")

        #format axes
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()])
        # ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks()])

    #hide unused axis
    unused = len(axs.flat) - len(fields)
    if unused > 0:
        for a in  axs.flat[-unused:]:
            a.axis('off')  

    plt.tight_layout()
    plt.show()

def plot_bar_prob(fields,df,w=20,h=7):
    """ Plot histogram for each given field.
    
        Params
        ------
        fields: list
                The list of columns we want to plot.
                
        df:     DataFrame
                Our Dataset.
    
    """
    # sns.set_color_codes("muted")

    nrows = int(math.ceil(len(fields) / 4))

    fig, axs = plt.subplots(nrows, 4,figsize=(w,int(nrows * h)))

    for ax, i in zip(axs.flat, fields):
        sns.barplot(x=i, y="TARGET", data=df, palette="muted", ax=ax)
        ax.set_ylabel("Default probability")

        
        # ax.set_yticklabels(['{:,.2f}'.format(y) for y in ax.get_yticks()])
        # ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks()])

    #hide unused axis
    unused = len(axs.flat) - len(fields)
    if unused > 0:
        for a in  axs.flat[-unused:]:
            a.axis('off')  

    plt.tight_layout()
    plt.show()


def plot_hbar_prob(fields,df,w=20,h=7):
    """ Plot histogram for each given field.
    
        Params
        ------
        fields: list
                The list of columns we want to plot.
                
        df:     DataFrame
                Our Dataset.
    
    """
    # sns.set_color_codes("muted")

    nrows = int(math.ceil(len(fields) / 4))

    fig, axs = plt.subplots(nrows, 4,figsize=(w,int(nrows * h)))

    for ax, i in zip(axs.flat, fields):
        sns.barplot(y=i, x="TARGET", data=df, palette="muted", ax=ax)
        ax.set_xlabel("Default probability")

        
        # ax.set_yticklabels(['{:,.2f}'.format(y) for y in ax.get_yticks()])
        # ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks()])

    #hide unused axis
    unused = len(axs.flat) - len(fields)
    if unused > 0:
        for a in  axs.flat[-unused:]:
            a.axis('off')  

    plt.tight_layout()
    plt.show()


def plot_hist(fields,df, hist=True,kde=False,rug=False):
    """ Plot histogram for each given field.
    
        Params
        ------
        fields: list
                The list of columns we want to plot.
                
        df:     DataFrame
                Our Dataset.
    
    """
    nrows = int(math.ceil(len(fields) / 4))

    fig, axs = plt.subplots(nrows, 4,figsize=(20,int(nrows * 7)))

    for ax, i in zip(axs.flat, fields):
        sns.distplot(df[df[i].notna()][i]  ,kde=kde, rug=rug,hist=hist ,ax=ax)

        #format axes
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks()])

        #rotate x axis
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    #hide unused axis
    unused = len(axs.flat) - len(fields)
    if unused > 0:
        for a in  axs.flat[-unused:]:
            a.axis('off')  

    plt.tight_layout()
    plt.show()

def plot_hist_target(fields,df):
    """ Plot histogram for each given field 
        by target
    
        Params
        ------
        fields: list
                The list of columns we want to plot.
                
        df:     DataFrame
                Our Dataset.
    
    """

    #filter the dataset by non-default and default
    non_default = df[df['TARGET']==0] 
    default = df[df['TARGET']==1]

    nrows = int(math.ceil(len(fields) / 4))

    fig, axs = plt.subplots(nrows, 4,figsize=(20,int(nrows * 7)))

    for ax, i in zip(axs.flat, fields):
        sns.distplot(non_default[i],hist=False ,kde=True,color="r", rug=False , label='Non-default',ax=ax)
        sns.distplot(default[i],hist=False ,kde=True,color="g", rug=False , label='Default',ax=ax)

        #format axes
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()])
        ax.set_xticklabels(['{:,.0f}'.format(x) for x in ax.get_xticks()])

        #rotate x axis
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        #add lengend
        ax.legend(ncol=2, loc="lower center", frameon=True)

    #hide unused axis
    unused = len(axs.flat) - len(fields)
    if unused > 0:
        for a in  axs.flat[-unused:]:
            a.axis('off')  

    plt.tight_layout()
    plt.show()


def root_path_pre_processed(file_name):
    """ the base path where the pre-processed datasets are located."""
    
    return r"data_pre_processed\\{}".format( file_name)

def get_pre_processed_combined():
    """ Return all the pre-processed datasets
        as one dataframe
    """

    #the applications pre-processed continuous data type dataset 
    app_cont_df = pd.read_csv(root_path_pre_processed('1_1_applications_continuous.csv'))

    #convert fields to int64
    cat1 = ['CNT_FAM_MEMBERS','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']

    for c in cat1:
        app_cont_df[c]  = app_cont_df[c].astype('int64')

    #the applications pre-processed integer data type dataset   
    app_int_df = pd.read_csv(root_path_pre_processed('1_2_applications_integer_types.csv'))
   
   # exclude TARGET and DATASET fields
    cols_int = [f for f in app_int_df.columns if f not in ['TARGET','DATASET']]  

    #merge the continuous and integer datasets
    df = app_cont_df.merge(app_int_df[cols_int],
                            how='left', 
                            on='SK_ID_CURR',
                            validate='one_to_one')


    #import the string applications data types
    app_string_df = pd.read_csv(root_path_pre_processed('1_3_applications_string_types.csv'))

    # exclude TARGET and DATASET fields
    cols_str = [f for f in app_string_df.columns if f not in ['TARGET','DATASET']]  

    #merge the string types
    df = df.merge(app_string_df[cols_str],
                            how='left', 
                            on='SK_ID_CURR',
                            validate='one_to_one')

    #import bureau pre-processed dataset
    bureau_df = pd.read_csv(root_path_pre_processed('1_4_bureau.csv'))
    
    #merge bureau 
    df = df.merge(bureau_df,
                    how='left', 
                    on='SK_ID_CURR',
                    validate='one_to_one')

    bureau_int_f = ['BUREAU_COUNT_GROUP',
                    'BUREAU_OVERDUE',
                    'BUREAU_DEBT_OVER_50%',
                    'BUREAU_DEBT_OVER_75%',
                    'BUREAU_DEBT_OVER_100%',
                    'BUREAU_ACTIVE',
                    'BUREAU_CLOSED',
                    'BUREAU_ADVERSE',
                    'BUREAU_CREDIT_TYPE_COUNT']

    for c in bureau_int_f:
        df[c].fillna(0,inplace=True)
        df[c]  = df[c].astype('float64')
    
    bureau_float_f = [f for f in bureau_df.columns if f !='SK_ID_CURR' and f not in bureau_int_f ]
    for c in bureau_float_f:
        df[c].fillna(0,inplace=True)
    
    #import pos balances pre-processed dataset
    pos_bal_df = pd.read_csv(root_path_pre_processed('1_5_pos_balances.csv'))

    #merge pos balance dataset
    df = df.merge(pos_bal_df,
                    how='left', 
                    on='SK_ID_CURR',
                    validate='one_to_one')

    #fill na and convert column to int64
    pos_cols_int = ['POS_BAL_COMPLETED_COUNT', 'POS_BAL_OVERDUE_ACTIVE_COUNT']
    for c in pos_cols_int:
        df[c].fillna(0,inplace=True)
        df[c]  = df[c].astype('float64')

    pos_cols_float = [f for f in pos_bal_df.columns if f != 'SK_ID_CURR' and f not in pos_cols_int]
    
    for c in pos_cols_float:
        df[c].fillna(0,inplace=True)
    
    #import previous applications pre-processed dataset
    pre_app_df = pd.read_csv(root_path_pre_processed('1_7_previous_applications.csv'))

    #merge previous applications dataset
    df = df.merge(pre_app_df,
                    how='left', 
                    on='SK_ID_CURR',
                    validate='one_to_one')

    #fill na and convert column to int64
    pre_app_cols = [f for f in pre_app_df.columns if f != 'SK_ID_CURR']

    for c in pre_app_cols:
        df[c].fillna(0,inplace=True)
        df[c]  = df[c].astype('float64')

    instll_df = pd.read_csv(root_path_pre_processed('1_8_installment_payments.csv'))
    df = df.merge(instll_df,
                    how='left', 
                    on='SK_ID_CURR',
                    validate='one_to_one')

    #handle installment na
    df['INSTLL_PAY_PAYMENT_GRADE'].fillna('none',inplace=True)
    df['TIMELY_PERCENT'].fillna(0,inplace=True)

    #meget credit card balances
    cc_df = pd.read_csv(root_path_pre_processed('1_6_cc_balances.csv'))

    df = df.merge(cc_df,
                    how='left', 
                    on='SK_ID_CURR',
                    validate='one_to_one')

    #fill na 
    cc_cols = [f for f in cc_df.columns if f != 'SK_ID_CURR']
    for c in cc_cols:
        df[c].fillna(0,inplace=True)
    
    return df


class NormData(object):
    """
        Class to normalize and one-hot-encode int and string fields.

        Params
        ------

        df: DataFrame
              Dataset that contains the combined pre-processed datasets.

    """
    def __init__(self,df, scaler):

        self.data = df.copy()

        self.scaler = scaler
        
        #create ABT
        self.abt = ABT(self.data)

        #normalize float features
        self.normalize_floats()
        print(">> Normalized float features.")

        org_count = len(self.data.columns)
        print(">> Column count before one hot encoding: {}.".format(org_count))

        #integer one-hot-encoding
        self.data = pd.get_dummies(self.data, columns=self.abt.int_features)
        print(">> One-hot-encoded integer features.")

        #lets one-hot encode the string fields
        self.one_hot_string()
        print(">> One-hot-encoded string features.")

        print(">> Column count after one hot encoding: {}.".format( len(self.data.columns)  ))
        print(">> Total columns added: {}.".format( len(self.data.columns) - org_count))
    
    def normalize_floats(self):
        """ Standarize float features. 
        """

        for feature in self.abt.float_features:
            #first normalize training dataset
            #get ids of training datasets
            ids_train = self.data_train[feature].index.values
            
            #perform standarization
            vals_train = self.data_train[feature].values.reshape(-1, 1)
            
            scaler = self.scaler.fit(vals_train)
            # scaler = StandardScaler().fit(vals_train)
            x_train = scaler.transform(vals_train)

            #update training values
            X_update_train = pd.DataFrame(x_train.flatten(), columns=[feature], index=ids_train)
            self.data.update(X_update_train)

            #now update test set using lmbda from training set
            ids_test = self.data_test[feature].index.values
            vals_test = self.data_test[feature].values.reshape(-1, 1)
            x_test = scaler.transform(vals_test)

            #update training values
            X_update_test = pd.DataFrame(x_test.flatten(), columns=[feature], index=ids_test)
            self.data.update(X_update_test)


    def one_hot_string(self):
        ''' One hot encode all the string fields.'''

        drop_features =[]
        one_hot_features = []
        for f in  self.abt.other_features:
            #get unique values from feature
            idx = self.data.groupby(f).size().index
            mapper = {f:i for i,f in enumerate(idx)}
            
            #lets create new map fields
            one_hot = f + "_MAP"
            self.data[one_hot] = self.data[f].map(mapper)

            drop_features.append(f)
            one_hot_features.append(one_hot)

        #lets drop original features 
        self.data.drop(drop_features,axis=1,inplace=True)

         #one-hot-encoding
        self.data = pd.get_dummies(self.data, columns=one_hot_features)

    @property
    def data_train(self):
        ''' Get training dataset '''
        return self.data[self.data['DATASET']=='train'] 

    @property
    def data_train_clean(self):
        ''' Get training dataset excluding  '''
        return self.data[self.data['DATASET']=='train'].drop(['DATASET'],axis=1) 

    @property
    def data_test(self):
        ''' Get testing dataset '''
        return self.data[self.data['DATASET']=='test'] 

    @property
    def data_test_clean(self):
        ''' Get testing dataset '''
        return self.data[self.data['DATASET']=='test'].drop(['TARGET','DATASET'],axis=1) 

def stratified_dataset(df,test_size,strat_field ='EXT_SOURCE_1' ):
    """ Perform stratification and return test index"""
    
    df_work = df.dropna(subset=[strat_field]).copy()
    
    #create stratifying field
    df_work['STRAT_FIELD'] = df_work[strat_field].apply(lambda x: int( round((x*10))) )
    
    # split between target value 1 and 0
    non_defaults = df_work[df_work['TARGET']==0].copy().reset_index(drop=True)
    defaults = df_work[df_work['TARGET']==1].copy().reset_index(drop=True)
    
    X = non_defaults.drop(['STRAT_FIELD'], axis=1).values
    y = non_defaults['STRAT_FIELD'].values
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=54)
    splits = list(sss.split(X, y))
    
    new_df = non_defaults.iloc[splits[0][1]].copy()
    new_df = new_df.append(defaults).reset_index(drop=True)
    
    return new_df

def predict_nan(predict,features, df):
    """ Predict the value of na for given feature"""
    #get X target for predct 
    temp_predict_df = df[df[predict].isna()]
    
    X_predict = temp_predict_df.dropna(subset=features).copy()
    
    X_predict_input = X_predict[features].values
    X_predict_ids = X_predict.index.values
    
    #lets filter only training dataset
    df_train = df[df['DATASET']=='train']
    
    #let get prediction training set
    
    X_df = df_train.dropna(subset=[predict] + features).copy()
    X_df = stratified_dataset(X_df,0.09)
    
    train_df = X_df
    X = train_df[features].values
    y = train_df[predict].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=54)
    
    model = LGBMRegressor(random_state = 54)
    model.fit(X_train, y_train ,eval_metric= 'r2',eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=-1)
    pred = model.predict(X_test,num_iteration=model.best_iteration_)

    # calculate mse
    mse = mean_squared_error(y_test, pred)
    print("MSE: {:.4f}".format(mse)) 
    
     #update training values
    update_pred = model.predict(X_predict_input ,num_iteration=model.best_iteration_)
    pred_update = pd.DataFrame(update_pred.flatten(), columns=[predict], index=X_predict_ids)
    df.update(pred_update)