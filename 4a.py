#Step 1
import pandas as pd
#Step 2
df = pd.read_csv('ch1ex1.csv')
points=df.values
#Step 3
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3)
model.fit(points)
labels=model.predict(points)
#Step 4
import matplotlib.pyplot as plt
xs = points[:,0]
ys = points[:,1]
plt.scatter(xs, ys, c=labels)
plt.show()

centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
plt.scatter(xs, ys, c=labels)
plt.scatter(centroids_x, centroids_y, marker='X', s=200)
plt.show()
