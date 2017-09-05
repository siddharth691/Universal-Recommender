#importing dependencies
import numpy as np
import pandas as pd
import logging

class urecommend:
    """
    This is a mathematical implementation of Universal recommender by actionml
    For more information visit: https://www.slideshare.net/pferrel/unified-recommender-39986309
    It saves the log file ('urecommend_log.log') in the specified input location

    Parameters
    -----------
    params: meta data about the recommender (dictionary)
            keys: 'eventNames': list of events
                  
                  'primaryEvent': zero indexed location of primary event in the eventNames list (int)
                  
                  'algorithm': (dictionary)
                               keys: 'name': 'Universal Recommender' (string)
                                     'no_recommendations': number of recommendations to return (int)
                                     'time_dependency': Include time factor or not (Boolean (True/False))
                  
                  'log_path': path where log file need to be saved (string)
                  'ipython_notebook': True if calling this function in a ipython notebook (boolean)
    Methods
    --------

    urecommend.fit(X)
    urecommend.predict(user_history)

    """

    
    def __init__(self, params):

        self.eventNames = params['eventNames']
        self.primaryEvent = params['primaryEvent']
        self.no_recommendations = params['algorithm']['no_recommendations']
        self.log_path = params['log_path']
        self.ipython = params['ipython_notebook']
        
        if not isinstance(self.eventNames, list):
            raise TypeError('eventNames should be of type list')
        
        if not isinstance(self.primaryEvent, int):
            raise TypeError('primaryEvent should be of type int')
        
        if not isinstance(self.no_recommendations, int):
            raise TypeError('no_recommendations should be of type int')

        if not isinstance(self.log_path, object):
            raise TypeError('log_path should be of type string')
        
        if not isinstance(self.ipython, bool):
        	raise TypeError('ipython_notebook should be of type boolean')

        if(self.ipython == True):
        	import imp
        	import logging
        	imp.reload(logging)

        logging.basicConfig(filename= self.log_path.rstrip('//')+'/urecommend_log.log',format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
        
        
    def fit(self, X):
        """
        This function trains the universal recommender
        
        Parameters
        -----------
        X: (dataframe with minimum three columns)('visitorid','event','itemid')
           each data point is an event by visitorid on the itemid
           visitorid, itemid are integers
           event is string

        """
        
        self.X = X
        
        #Check if input dataframe has all required columns or not
        if isinstance(self.X, pd.DataFrame):
            if(not set(['visitorid','event','itemid']).issubset(set(self.X.columns.values))):
                raise ValueError('Incomplete input dataframe')
        else:
            raise TypeError('Input is not a dataframe')
        
        #Check if params events are not equal to input dataframe unique events
        if(set(self.X.event.unique())!=set(self.eventNames)):
            if(len(self.X.event.unique())>len(self.eventNames)):
                raise ValueError('Input dataframes have more unique events than specified in input params')
            elif(len(self.X.event.unique())<len(self.eventNames)):
                diff_elem = set(self.eventNames) - set(self.X.event.unique())
                logging.info('Input dataframe have no datapoints for '+ ' and '.join(diff_elem))
        
        #Create history matrix
        self.create_history_matrix()
        logging.info('History matrices created')
        
        #Create union history matrix
        self.create_union_history_matrix()
        logging.info('Union of history matrices created')
        
        #Calculating coocurrence and cross-coocurrence matrices
        self.create_coocurrence_matrix()
        logging.info('Coocurrence matrices created')
        logging.info('u-recommender trained to the data')
        
    def predict(self, user_history):
        
        """
        This function predicts the recommended products
        
        Parameters
        -----------
        user_history:(dictionary of length equal to number of events in params)
                     keys will be event name
                     and values will be a 2D list of item_id and count pairs
                     if there are no items for any event, value for that event will be None

        Output
        -------
        recommendation: (list of itemid)
        """
        #Check if user_history has more or less items than total item
        if(not set(self.eventNames)==set(user_history.keys())):
            raise ValueError('All events as mentioned in params are not present in user_history')
        
        user_llr = np.zeros((len(self.coocurrence[list(self.coocurrence.keys())[0]]),1))
        
        for event,matrix in user_history.items():
                        
            user = pd.DataFrame(0,index = self.coocurrence[list(self.coocurrence.keys())[0]].index, columns =['user'])
            if(matrix is not None):
                
                #Check if input user_history is list of list
                if(not isinstance(matrix, list)):
                    raise ValueError('Values of user_history '+event+' key should be list of list')
                
                matrix = dict(matrix)

                for index,series in user.iterrows():
                    if(index in list(matrix.keys())):
                        user.loc[index,'user']=matrix[index]
        
            user_llr += self.return_recommended_llr(self.coocurrence[event], user)
        
        user_llr = pd.DataFrame(user_llr, index=self.coocurrence[list(self.coocurrence.keys())[0]].index, columns =['llr'])

        logging.info('predicted for given user')

        return list(user_llr.sort_values('llr', axis=0, ascending=False).index)[:self.no_recommendations]
        
    def create_history_matrix(self):
        
        #Initializing empty dictionary
        self.history = {}
        
        for eventName in self.eventNames:
            
            #Taking out only event specific columns
            self.history[eventName] = self.X.loc[self.X.event == eventName]
            
            #Taking out required columns
            self.history[eventName] = self.history[eventName].loc[:,['visitorid','itemid']]
            
            #Making visitorid as index
            self.history[eventName].index = self.history[eventName].visitorid
            self.history[eventName].drop('visitorid', axis=1, inplace =True)
            
            #creating dummy variable for each itemid
            self.history[eventName] = pd.get_dummies(self.history[eventName], columns=['itemid'])
            
            #adding count when user participated in an event with same item more than once
            self.history[eventName] = self.history[eventName].groupby([self.history[eventName].index])[self.history[eventName].filter(regex='itemid_.*').columns].sum()
        
        
    def create_union_history_matrix(self):
        
        #Calculating index and column union
        index_union = self.history[self.eventNames[0]].index
        column_union = self.history[self.eventNames[0]].columns
        
        for eventName in self.eventNames[1:]:
            index_union = index_union.union(self.history[eventName].index)
            
            column_union = column_union.union(self.history[eventName].columns)
        
        #Creating union history event dataframe using index and column union
        for eventName in self.eventNames:
            
            self.history[eventName] = self.union_dataframe(self.history[eventName], index_union, column_union)
    
    def create_coocurrence_matrix(self):
        
        self.coocurrence = {}
        
        self.coocurrence[self.eventNames[self.primaryEvent]] = self.calc_cooccurence_matrix(self.history[self.eventNames[self.primaryEvent]])
        
        logging.debug('cooccurence matrix of primary event - %s is calculated',self.eventNames[self.primaryEvent])
        
        for eventName in self.eventNames:
            if(eventName!= self.eventNames[self.primaryEvent]):
                self.coocurrence[eventName] = self.calc_cross_coocurence_matrix(self.history[self.eventNames[self.primaryEvent]], self.history[eventName])
                logging.debug('cross-cooccurence matrix of event - %s is calculated',eventName)
                
    def union_dataframe(self, history_event, ind_union, col_union):
    
        history_event = history_event.copy()
        
        index_event_df = pd.DataFrame(0, index=ind_union.difference(history_event.index), columns= history_event.columns)
        
        history_event = pd.concat([history_event, index_event_df], axis=0)
        
        col_event_df = pd.DataFrame(0,index=history_event.index, columns=col_union.difference(history_event.columns))
        
        history_event= pd.concat([history_event, col_event_df], axis=1)
        
        #Sort rows and columns for uniformity
        history_event = history_event.reindex_axis(sorted(history_event.columns), axis=1).sort_index()
        return history_event
    
    def calc_cooccurence_matrix(self, history_event):
        
        coo_matrix = pd.DataFrame(0, index = history_event.columns, columns = history_event.columns)
        for item1_index in range(len(history_event.columns)):
            for item2_index in range(len(history_event.columns)):
                
                item1 = np.array(history_event.iloc[:,item1_index].values)
                
                item2 = np.array(history_event.iloc[:,item2_index].values)
                
                k11,k12,k21,k22 = self.calc_counts_row(item1,item2)
                
                llr = self.llr_2x2(k11,k12,k21,k22)
                
                coo_matrix.iloc[item1_index,item2_index] = llr
        return coo_matrix
    
    def calc_cross_coocurence_matrix(self, primary_history, secondary_event):
        coo_matrix = pd.DataFrame(0, index = primary_history.columns, columns = secondary_event.columns)
        for item1_index in range(len(primary_history.columns)):
            for item2_index in range(len(secondary_event.columns)):
                
                item1 = np.array(primary_history.iloc[:,item1_index].values)
                
                item2 = np.array(secondary_event.iloc[:,item2_index].values)
                
                k11,k12,k21,k22 = self.calc_counts_row(item1,item2)
                
                llr = self.llr_2x2(k11,k12,k21,k22)
                
                coo_matrix.iloc[item1_index,item2_index] =llr
        return coo_matrix
    
    def calc_counts_row(self, item1, item2):
        new_item = np.concatenate((item1.reshape(-1,1),item2.reshape(-1,1)), axis=1)
        if((np.any(new_item[:,1]<0)==True)|(np.any(new_item[:,0]<0)==True)):
            raise ValueError('History matrix has negative element')
        k22 = len(new_item[(new_item[:,0]==0)&(new_item[:,1]==0)])
        
        k21 = len(new_item[(new_item[:,0]==0)&(new_item[:,1]!=0)])
        
        k12 = len(new_item[(new_item[:,0]!=0)&(new_item[:,1]==0)])

        k11 = len(new_item[(new_item[:,0]!=0)&(new_item[:,1]!=0)])

        return k11,k12,k21,k22
    
    def llr_2x2(self, k11, k12, k21, k22):
        '''Special case of llr with a 2x2 table'''
        return 2 * (self.denormEntropy([k11+k12, k21+k22]) +
                    self.denormEntropy([k11+k21, k12+k22]) -
                    self.denormEntropy([k11, k12, k21, k22]))
    
    def denormEntropy(self,counts):
        '''Computes the entropy of a list of counts scaled by the sum of the counts. 
            If the inputs sum to one, this is just the normal definition of entropy'''
        lg = np.log(np.divide(counts,float(np.sum(counts))))
        
        lg[lg==-np.inf]=0
        return -np.sum(counts*lg)
    
    def return_recommended_llr(self, coo_dataframe, user_history):
        
        coo_matrix = coo_dataframe.as_matrix()
        
        user = np.array(user_history.values).reshape(-1,1)
        
        user_llr = np.dot(coo_matrix, user)
        return user_llr
