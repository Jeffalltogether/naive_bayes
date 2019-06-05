# MIT License

# Copyright (c) 2019 Jeffrey Thatcher

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pandas as pd
import numpy as np


def generate_CLG(data, outcome_var = 'y', cont_var = 'x'):
    # separate the data
    sub_df = data[[outcome_var, cont_var]]

    # compute levels of outcome variable
    y_levels = np.unique(sub_df[outcome_var])
    y_levels = y_levels[~np.isnan(y_levels)]

    # compute the means and standard deviations for each continuous variable

    CLG_means = sub_df.groupby([outcome_var], as_index=False).mean()
    CLG_stds = sub_df.groupby([outcome_var], as_index=False).std()

    CLG_params = {}
    for l_y in y_levels:
        CLG_params.update({str(float(l_y)): []})

    for row_m, row_s in zip(CLG_means.values, CLG_stds.values):
        CLG_params.update({str(row_m[0]): [row_m[1], row_s[1]]})

    return CLG_params

def generate_CPD(data, outcome_var = 'y', nom_var = 'x'):
    import copy

    sub_df = data[[outcome_var, nom_var]]

    y_levels = np.unique(sub_df[outcome_var])
    y_levels = y_levels[~np.isnan(y_levels)]

    x_levels = np.unique(sub_df[nom_var])
    x_levels = x_levels[~np.isnan(x_levels)]


    temp_entry = {}
    for l_x in x_levels:
        temp = {str(l_x):0.0}
        temp = copy.deepcopy(temp)
        temp_entry.update(temp)

    obs = {}
    temp_CPD = {}
    for l_y in y_levels:
        obs.update({str(l_y): []})
        for l_x in x_levels:
            temp = {str(l_y): temp_entry}
            temp = copy.deepcopy(temp)
            temp_CPD.update(temp)

    CPD = copy.deepcopy(temp_CPD)


    for l_y in y_levels:
        for row in sub_df.values:
            if np.isnan(row).any():
                continue
            elif row[0] == l_y:
                obs[str(l_y)].append(row[1])

    for l_y in y_levels:
        for l_x in x_levels:
            total = np.sum(np.array(obs[str(l_y)]) == l_x)

            if total == 0.0:
                total = 1.0

            num = len(obs[str(l_y)])

            p = float(total)/num

            CPD[str(l_y)][str(l_x)] = p

    return CPD

def CLG_calculateProbability(x, CPD, cls):
    '''
    Here is our function for calculating the probability that an unknown entry belongs to a certain class
    for our Conditional Linear Gaussian
    '''
    import scipy.stats

    if np.isnan(x):
        return np.nan

    else:
        x = float(x)
        cls = str(float(cls))

        mean = CPD[cls][0]
        stdev = CPD[cls][1]

        p = 2.0 * scipy.stats.norm.cdf(-1.0 * np.abs(x-mean), 0,stdev)

        return p

def CPT_calculateProbability(x, CPD, cls):
    '''
    Here is our function for calculating the probability that an unknown entry belongs to a certain class
    for our Conditional Probability Distributions
    '''

    if np.isnan(x):
        return np.nan

    else:
        x = str(float(x))
        cls = str(float(cls))

        try:
            p = CPD[cls][x]

        except KeyError:
            p = np.nan

        return p


class NB_Classifier(object):
    import numpy as np
    import pandas as pd

    def __init__(self, training_df, continuous_variables = [], multinomial_variables = [], outcome_variable = 'y'):
        self.data = training_df
        self.outcome_var = outcome_variable
        self.cont_vars = continuous_variables
        self.nom_vars = multinomial_variables
        
        
    def train_prevalence(self):
        sub_df = self.data[self.outcome_var].dropna()

        y_levels = np.unique(sub_df.values)
        y_levels = y_levels[~np.isnan(y_levels)]

        prevalence_params = {}
        for l_y in y_levels:
            prev = sum(sub_df.values == l_y) / float(len(sub_df))
            prevalence_params.update({str(float(l_y)):prev})

        self.prevalence_params = prevalence_params
        
        return
    
    def train_continuous_variables(self):

        CLG_list = {}
        for var in self.cont_vars:
            CLG = generate_CLG(self.data, self.outcome_var, var)
            CLG_list.update({str(var):CLG})

        self.CLG_list = CLG_list
        
        return
    
    def train_multinomial_variables(self):

        CPD_list = {}
        for var in self.nom_vars:
            CPD = generate_CPD(self.data, self.outcome_var, var)
            CPD_list.update({str(var):CPD})

        self.CPD_list = CPD_list
        
        return

        
    def inference(self, test_data = pd.DataFrame([])):
        # grab test data if given
        if not test_data.empty:
            print 'Inferencing on Test Data'
            self.data = test_data
        else:
            print 'Inferencing on Training Data'
        
        # drop the outcome variable from the input data frame
        input_data = self.data.drop(self.outcome_var, axis=1)

        # get the levels of the outcome variable exclusing the nan's
        y_levels = np.unique(self.data[self.outcome_var])
        y_levels = y_levels[~np.isnan(y_levels)]

        # compute Probability of each Class in outcome_var
        class_probabilities = {}
        for cls in y_levels:
            cls = str(float(cls))

            # generate aray of conditional probabilitis for each feature 
            conditional_probs = []
            for var in input_data.columns:
                # compute for continuous variables based on Conditional Linear Gaussian 
                if var in self.cont_vars:

                    p = np.vectorize(CLG_calculateProbability, excluded=['CPD'])(input_data[var], self.CLG_list[var], cls)

                    conditional_probs.append(p)

                # compute for multinomial variables based on Tabular Conditional Probability Distribution 
                if var in self.nom_vars:

                    p = np.vectorize(CPT_calculateProbability, excluded=['CPD'])(input_data[var], self.CPD_list[var], cls)

                    conditional_probs.append(p)

            # convert to np.array
            conditional_probs = np.array(conditional_probs)

            # temporarily replace nan's with 1.0
            conditional_probs[np.isnan(conditional_probs)]=1.0

            # multiply conditional proability for each feature from each observation 
            cls_probs = conditional_probs[0]
            for i in range(len(conditional_probs)-1):
                cls_probs *= conditional_probs[i+1]

            # if any are exactly 1.0 we know that all entries for that observation were nan's, we can replace these with nan
            cls_probs[cls_probs==1.0]=np.nan

            # multiply the resuling probabilities with the prevalence
            cls_probs = cls_probs * self.prevalence_params[str(float(cls))]

            # add to dictionary with result for each class
            class_probabilities.update({cls:cls_probs})

        self.class_probabilities = class_probabilities
        
        # compute the predicted class
        probability_matrix = []
        for key,values in self.class_probabilities.items():
            probability_matrix.append(values)

        probability_matrix = np.array(probability_matrix)

        argmax = np.array([i for i in np.argmax(self.class_probabilities.values(), axis=0)])
        prediction = [self.class_probabilities.keys()[i] for i in argmax]

        self.prediction = prediction

        return

    def compute_results(self):
        # compute the actual class
        ground_truth = np.array([str(float(i)) for i in self.data[self.outcome_var].values])

        outcome = []
        for i in range(len((self.prediction))):

            if self.prediction[i] == 'nan':
                comp = np.nan
            elif ground_truth[i] == 'nan':
                comp = np.nan
            else:
                comp = self.prediction[i] == ground_truth[i]

            outcome.append(comp)

        # compute accuracy
        outcome_array = np.array(outcome)

        outcome_array = outcome_array[~np.isnan(outcome_array)]    

        accuracy = float(sum(outcome_array)) / len(outcome_array)
        print 'algorithm accuracy: %.5f' %(accuracy)

        self.ground_truth = ground_truth
        self.outcome = outcome
        self.accuracy = accuracy
        
        return


if __name__ == "__main__":


	y = np.concatenate((np.zeros(10),np.ones(10)), axis=0)

	x1 = np.random.normal(loc=1, scale=0.2, size=20)
	x2 = y + np.random.normal(loc=0.0, scale=0.2, size=20)
	x3 = np.concatenate((np.random.binomial(n=1, p=0.85, size = 10), np.random.binomial(n=1, p=0.15, size = 10)), axis = 0)
	x4 = np.random.binomial(n=1, p=0.5, size = 20)

	train_df = pd.DataFrame()
	train_df['y'] = y
	train_df['x1'] = x1
	train_df['x2'] = x2
	train_df['x3'] = x3
	train_df['x4'] = x4
	train_df



	training_df = train_df
	continuous_variables = ['x1','x2']
	multinomial_variables = ['x3','x4']
	outcome_variable = 'y'

	my_NB = NB_Classifier(training_df, continuous_variables, multinomial_variables, outcome_variable)
	my_NB.train_prevalence()
	my_NB.train_continuous_variables()
	my_NB.train_multinomial_variables()
	my_NB.compute_conditional_probs()
	my_NB.compute_results()



	y = np.concatenate((np.zeros(10),np.ones(10)), axis=0)
	x1 = np.zeros(20)
	x2 = np.zeros(20)
	x3 = np.zeros(20)
	x4 = np.zeros(20)

	test_df = pd.DataFrame()
	test_df['y'] = y
	test_df['x1'] = x1
	test_df['x2'] = x2
	test_df['x3'] = x3
	test_df['x4'] = x4
	test_df

	my_NB.compute_conditional_probs(test_df)
	my_NB.compute_results()

