import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


dataframe=pd.read_csv('Challenge_dataset1.txt', header=None)  #fwf is fixed with formatted  2 d array
x=dataframe[[0]]
y=dataframe[[1]]

# Classifiers
# using the default values for all the hyperparameters  s = s.replace(',', ' ')
body_reg=linear_model.Ridge()

#train the models
body_reg.fit(x,y)

#using matplotlib library plot 

plt.scatter(x,y)
plt.plot(x,body_reg.predict(x))
plt.show()