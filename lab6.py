import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

games = pandas.read_csv("/Users/diego/Dropbox/Tutoring 722/Labs/Lab 6/board_games.csv")
print(games.columns)
print(games.shape)

plt.hist(games["average_rating"])
plt.show()

games[games["average_rating"] == 0]

print(games[games["average_rating"] == 0].iloc[0])
print(games[games["average_rating"] > 0].iloc[0])

games = games[games["average_rating"] > 0]
games = games.dropna(axis=0)

kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = games._get_numeric_data()
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_

pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

games.corr()["average_rating"]

columns = games.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name"]]
target = ["average_rating"]

train = games.sample(frac=0.8, random_state=1)
test = games.loc[~games.index.isin(train.index)]
print(train.shape)
print(test.shape)

model = LinearRegression()
model.fit(train[columns], train[target])

predictions = model.predict(test[columns])
mean_squared_error(predictions, test[target])

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
model.fit(train[columns], train[target])
predictions = model.predict(test[columns])
mean_squared_error(predictions, test[target])