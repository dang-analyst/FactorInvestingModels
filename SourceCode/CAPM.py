
from pandas import *
from numpy import *
import random
import matplotlib
import matplotlib.pyplot as plt 
import statsmodels.api as sm



# training test proportions
training_size = 0.75
test_size = 1.0 - training_size
divisor = 10.0

# data locations
dataLocations = read_csv('dataLocations.csv', encoding='utf-8')
dataFama3 = read_csv('Fama3.csv', encoding= 'utf-8')

# master_price_data
daily_data = read_csv(dataLocations['path'].iloc[1])
#print(daily_data)
master_price_data = DataFrame(index=daily_data['Date'], columns=dataLocations['ticker'])
for index, row in dataLocations.iterrows():
    daily_data = read_csv(row['path'])
    prices = daily_data.set_index('Date')['Close'].to_dict()
    prices_series = Series(prices)
    master_price_data[row['ticker']] = prices_series

master_price_data = master_price_data.sort_index()
master_price_data.index = to_datetime(master_price_data.index)
master_price_data = master_price_data.resample("W-MON").mean()
#print(master_price_data)

# calculate weekly returns
master_returns_data = master_price_data.pct_change(periods=1, fill_method= 'pad')
master_returns_data = master_returns_data.replace(NaN, 0)
dataFama3.index = master_returns_data.index
#print(master_returns_data)


# Finding Coefficients
Coeffiecients = DataFrame(columns= master_returns_data.columns)


X = dataFama3[['Mkt-RF']]
X = sm.add_constant(X)
Z = master_returns_data.columns
for item in Z:
    Y = master_returns_data[item] - dataFama3['RF']
    Fama_model = sm.OLS(Y,X).fit()
    Coeffiecients[item] = Fama_model.params
Coeffiecients.index = ['alpha','betaRM']

dataFama3 = dataFama3.set_index(master_returns_data.index)
master_returns_data["RM"] = dataFama3["Mkt-RF"]





# testing phase (divide into training/testing)
index_list = random.sample(range(0,len(master_price_data.index),int(divisor)), int(len(master_price_data.index) / divisor))
num_train = int(floor(training_size * len(master_price_data.index)))
reference_training_list = index_list[:int(num_train/divisor)]
reference_testing_list = index_list[int(num_train/divisor):]
training_list = []
training_list_2 = []
testing_list = []
testing_list_2 = []
for item in reference_training_list:
	for i in range (0,5):
		training_list.append(item + i)

for i in training_list:
    if i <= 2990:
        training_list_2.append(i)        

for item in reference_testing_list:
	for i in range (0,5):
		testing_list.append(item + i)

for i in testing_list:
    if i <= 2990:
        testing_list_2.append(i)  

training_returns_data = master_returns_data.iloc[training_list_2]
testing_returns_data = master_returns_data.iloc[testing_list_2]

#beta alpha, variance
alphas = Coeffiecients.loc['alpha'].transpose()
betaRM = Coeffiecients.loc['betaRM'].transpose()

#print(betas, alphas)

# error terms
alpha_ones = ones((1, len(training_returns_data.index)))
alphas_matrix = alphas.to_numpy().reshape(len(alphas.index), 1)
betas_matrix = betaRM.to_numpy().reshape(len(betaRM.index), 1)
Rm = training_returns_data['RM'].to_numpy().reshape(1, len(training_returns_data.index))



# model
predicted_returns_matrix = multiply(alphas_matrix, alpha_ones) + betas_matrix * Rm
predicted_returns_matrix = predicted_returns_matrix.transpose()
predicted_returns = DataFrame(data=predicted_returns_matrix, index=training_returns_data.index, columns=Coeffiecients.columns)


#Reomve RM, SMB, HML, RF from data
training_returns_data = training_returns_data.drop (['RM'] , axis =1)

epsilon_returns = training_returns_data.subtract(predicted_returns).mean()
epsilon_matrix = epsilon_returns.to_numpy().reshape(len(epsilon_returns.index), 1)
variance =  multiply(multiply(epsilon_matrix, epsilon_matrix), 1.0/(len(training_returns_data.index) - 2.0))

# testing phase
testing_ones = ones((1, len(testing_returns_data.index)))
Rm_testing = testing_returns_data['RM'].to_numpy().reshape(1, len(testing_returns_data.index))


testing_predicted_returns_matrix = multiply(alphas_matrix, testing_ones) + betas_matrix * Rm_testing + multiply(epsilon_matrix, testing_ones)
testing_predicted_returns_matrix = testing_predicted_returns_matrix.transpose()
testing_predicted_returns = DataFrame(data=testing_predicted_returns_matrix, index=testing_returns_data.index, columns=Coeffiecients.columns)

#Reomve RM, SMB, HML, RF from data
testing_returns_data = testing_returns_data.drop (['RM'] , axis =1)

differences = testing_returns_data.subtract(testing_predicted_returns)
difference_between_mean = testing_returns_data.subtract(testing_returns_data.mean())

# MSE and R2
ss_res = (differences ** 2).sum()
ss_tot = (difference_between_mean ** 2).sum()
r_squared = 1.0 - (ss_res / ss_tot)
mse_data = ss_res / (len(testing_returns_data.index))

summary = DataFrame(columns = ['alpha', 'beta', 'MSE', 'R2'])
summary['alpha'] = alphas
summary['beta'] = betaRM
summary['MSE'] = mse_data
summary['R2'] = r_squared
summary= summary.transpose()
