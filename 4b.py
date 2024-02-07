#Step 1
import pandas as pd
#Step 2
seeds_df = pd.read_csv('seeds-less-rows.csv')
varieties=list(seeds_df.pop("grain_variety"))
samples = seeds_df.values

#Step 3
from scipy.cluster.heirarchy import linkage,dendrogram
import matplotlib.pyplot as plt
mergings=linkage(samples,method=complete)
dendrogram(mergings,labels=varieties,leaf_rotation=90,leaf_font_size=8)
plt.show()

from scipy.cluster.heirarchy import FCluster
labels=fcluster(mergings,6,criterion='distance')

#Step 4
df=pd.Dataframe({"labels":labels,"varieties":varieties}]
ct-pd.crosstab(df["labels"],df["varieties"])

