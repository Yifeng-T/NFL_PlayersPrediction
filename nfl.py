import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
import numpy as np
from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack
from scipy import stats
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#read_in data frame
df = pd.read_csv("./nfl_players.csv")
print(df)
df_position = df["position"]
posi = set(df_position)
# total 5 positions:
# {'defensive end', 'defensive tackle', 'cornerback', 'quarterback', 'running back'}

#calculate mean of height and weight of each position
for position in posi:
    mean_height = sum(df[df["position"] == position]["height (in)"]) / \
        len(df[df["position"] == position]["height (in)"])
    print(f"{position} mean height is {mean_height}")

    mean_weight = sum(df[df["position"] == position]["weight (lb)"]) / \
        len(df[df["position"] == position]["weight (lb)"])
    print(f"{position} mean weight is {mean_weight}")

#predictoin function:
# given height and weight value: predict the position
def prediction(height, weight):
    #reconstruct the data frames
    df_info = df[["height (in)", "weight (lb)"]]
    X = np.array(df_info)
    Y = np.array(df_position)

    #find the correct model:
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, Y)
    LinearDiscriminantAnalysis()
    return(clf.predict([[height, weight]]))

# What is the best position for someone who is 5'10" and 155 lbs?
# 5'10" = 70''
print(prediction(70, 155)) #======>['cornerback']
print(prediction(77, 230)) #======>['quarterback']
print(prediction(90, 330)) #======>['defensive end']
print(prediction(60, 320)) #======>['defensive tackle']
print(prediction(66, 260)) #======>['running back']


# draw the classifier
df_graph = df 
#I dont know how to treat the string value for classfier, 
# so change position_name to corresponding index value
df_graph['position'] = df_graph['position'].replace(['defensive end','defensive tackle',\
    'cornerback','quarterback','running back'],[0, 1, 2,3, 4])

#rebuild the data frame needed
df_position_graph = df_graph["position"]
df_info_graph = df_graph[["height (in)", "weight (lb)"]]
posi_graph = set(df_position_graph)
#define the new X and Y
X_graph = np.array(df_info_graph)
Y_graph = np.array(df_position_graph)

# generate dataset

# define bounds of the domain
min1, max1 = X_graph[:, 0].min()-1, X_graph[:, 0].max()+1
min2, max2 = X_graph[:, 1].min()-1, X_graph[:, 1].max()+1
# define the x and y scale
x1grid = arange(min1, max1, 0.1)
x2grid = arange(min2, max2, 0.1)
# create all of the lines and rows of the grid
xx, yy = meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = hstack((r1,r2))
# define the model

model = LinearDiscriminantAnalysis()
model.fit(X_graph, Y_graph)
# make predictions for the grid
yhat = model.predict(grid)
# reshape the predictions back into a grid
zz = yhat.reshape(xx.shape)


# plot the grid of x, y and z values as a surface
pyplot.contourf(xx, yy, zz, cmap = 'Paired')
colour = ['#2d618e', '#8eab12', '#74050b', '#d1768f', '#f9a51b']
# create scatter plot for samples from each class
#rename the position
posii = list(posi_graph)
for i in range(len(posii)):
    if posii[i] == 0:
        name = 'defensive end'
    elif posii[i] ==1:
        name = 'defensive tackle'
    elif posii[i] ==2:
        name = 'cornerback'
    elif posii[i] ==3:
        name = 'quarterback'
    else:
        name = 'running back'
    #get row indexes for samples with this class
    row_ix = where(Y_graph == posii[i])
	#create scatter of these samples
    pyplot.scatter(X_graph[row_ix, 0], X_graph[row_ix, 1], label=name, color = colour[i])

# show the plot
pyplot.legend()
pyplot.xlabel("height", size=14)
pyplot.ylabel("weight", size=14)
plt.title('NFL Players Height&Weight position classifiers', size=16)
pyplot.savefig("classifier")
pyplot.show()
