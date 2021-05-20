import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ipl = pd.read_csv("ipldata")
# print(ipl['player_of_match'].value_counts()[0:10])
# print(ipl['player_of_match'].value_counts()[0:5])

plt.figure(figsize=(8,5))
plt.xlabel("No. Of Matches")
plt.ylabel("Players")
plt.bar(list(ipl['player_of_match'].value_counts()[0:5].keys()),list(ipl['player_of_match'].value_counts()[0:5]),color='g')
plt.show()