import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

match_df = pd.read_csv("ipldata")

# Most Wins in IPL
# temp=pd.DataFrame({"Winner":match_df['winner']})
# count_wins=temp.value_counts()
# labels=[X[0] for X in count_wins.keys()]
# bzr,ax=plt.subplots(figsize=(20,12))
# ax=plt.pie(x=count_wins, labels=labels)
# plt.title("Most Wins In IPl", fontsize=18)
# plt.show()
#





# toss = match_df.groupby(['season', 'toss_winner']).winner.value_counts().reset_index(name = 'count')
# toss['result'] = np.where(toss.toss_winner == toss.winner, 'won', 'lost')
# toss_result = toss.groupby(['season', 'toss_winner','result'])['count'].sum().reset_index()
#
# for x in range(2008, 2009, 1):
#     toss_result_x = toss_result[toss_result['season'] == x]
#     plot = sns.barplot(x="toss_winner", y="count", hue="result", data=toss_result_x)
#     plot.set_title('Matches won/lost by teams winning toss \nSeason ' +str(x))
#     #plot.set_xticklabels(rotation=30,labels=toss_result_x['toss_winner'])
#     plot.set_xticklabels(toss_result['toss_winner'], rotation=30)
#     plt.show()
# #     x+=1
# temp=pd.concat([match_df['umpire1'],match_df['umpire2']]).value_counts().sort_values(ascending=False)
#
# plt.figure(figsize=(20,5))
# Most_umpired =sns.barplot(x=temp.index, y=temp.values, alpha=0.9)
#
# plt.title('Favorite umpire')
# plt.ylabel('Count', fontsize=12)
# plt.xlabel('Name of the Umpire', fontsize=15)
# Most_umpired.set_xticklabels(rotation=90,labels=temp.index,fontsize=15)
# plt.show()
# #Most Played Venue
# match_df = pd.read_csv("ipldata")
# warnings.simplefilter(action='ignore', category=FutureWarning)
# sns.set_style("darkgrid")
# ls=match_df['venue'].value_counts().sort_values(ascending=False)
# plt.figure(figsize=(10,8))
# temp =sns.barplot(ls.index, ls.values, alpha=0.8)
# plt.title('MOST PLAYED VENUE')
# plt.ylabel('COUNT', fontsize=14)
# plt.xlabel('NAME OF THE STADIUMS', fontsize=15)
# temp.set_xticklabels(rotation=90,labels=ls.index,fontsize=5)
# plt.show()

#Stadium Wise Wining
# Chennai_stadium=match_df.loc[(match_df['venue']=='MA Chidambaram Stadium, Chepauk') ]
# Chennai_stadium_win_by_runs=Chennai_stadium[Chennai_stadium['win_by_runs']>0]
# slices=[len(Chennai_stadium_win_by_runs),len(Chennai_stadium)-len(Chennai_stadium_win_by_runs)]
# labels=['Batting first','Batting Second']
# plt.pie(slices,labels=labels,startangle=90,shadow=1,explode=(0,0.2),autopct='%4.2f%%',colors=['#11fcf4','#ef09aa'])
# plt.title('MA Chidambaram Stadium, Chepauk')
# plt.show()

team_encodings={
'Mumbai Indians':1,
'Kolkata Knight Riders':2,
'Royal Challengers Bangalore':3,
'Deccan Chargers':4,
'Chennai Super Kings':5,
'Rajasthan Royals':6,
'Delhi Daredevils':7,
'Gujarat Lions':8,
'Kings XI Punjab':9,
'Sunrisers Hyderabad':10,
'Rising Pune Supergiant':11,
'Kochi Tuskers Kerala':12,
'Pune Warriors':13,
'Delhi Capitals':14,
'Draw':15
}
team_encode_dict={
    'team1':team_encodings,
    'team2' :team_encodings,
    'toss_winner' :team_encodings,
    'winner' :team_encodings,
}
match_df.replace(team_encode_dict, inplace=True)
# print(match_df.head(10))
# print(match_df[match_df['winner'].isnull()==True])
match_df['winner'].fillna('Draw', inplace=True)
match_df.replace(team_encode_dict, inplace=True)
# match_df.info()
# print(match_df[match_df['team1'].isnull() == True])

# print(match_df['team2'])
# print(match_df[match_df['city'].isnull() == True])
match_df['city'].fillna('Dubai', inplace=True)
# match_df.info()
# print(match_df.describe())
# Dropping all the redundant columns
match_df = match_df[[ 'team1','team2','city','toss_decision','toss_winner','venue','winner']]
# print(match_df.head())
#looking at number of toss wins and match wins
toss_wins = match_df['winner'].value_counts(sort=True)
match_wins = match_df['winner'].value_counts(sort=True)

# for idx, val in toss_wins.iteritems():
#     print(f"{list(team_encode_dict['winner'].keys())[idx-1]} -> {toss_wins[idx]}")

#using the label encoder


ftr_list = ['city', 'toss_decision', 'venue']
encoder = LabelEncoder()
for ftr in ftr_list:
    match_df[ftr] = encoder.fit_transform(match_df[ftr])

    # print(encoder.classes_)

# print(match_df)
#splitting the data for training and testing
# r={}
# r=match_df['winner']
# print(r.unique())


train_df, test_df = train_test_split(match_df, test_size=0.2, random_state=42)
print(train_df.shape)
print(test_df.shape)


def print_model_scores(model, data, predictors, target):
    '''
    A generic function to generate the performance report of the
    model in question on the data passed to it using cross-validation

    Args:
        model: ML Model to be checked
        data: data on which the model needs to pe trained
        predictors: independent feature variable
        target: target variable
    '''


    model.fit(data[predictors],data[target])
    predictions = model.predict(data[predictors])
    accuracy = accuracy_score(predictions, data[target])
    # print('Accuracy : %s' % '{0:.2%}'.format(accuracy))
    scores = cross_val_score(model, data[predictors], data[target], scoring="neg_mean_squared_error", cv=5)
    # print('Cross-Validation Score :{}'.format(np.sqrt(-scores)))
    # print(f"Average RMSE: {np.sqrt(-scores).mean()}")
#
model = RandomForestClassifier(n_estimators=1000)
target_var = ['winner']
# print(match_df.head())
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
print_model_scores(model, match_df,predictor_var, target_var)
team1=input("Enter the First Team")
# team1='Mumbai Indians'
team2=input("Enter the Second Team")
# team2='Kolkata Knight Riders'
toss_winner=input("Enter the Team Who won the toss")
# toss_winner='Sunrisers Hyderabad'
inp = [team_encode_dict['team1'][team1],team_encode_dict['team2'][team2],'10',team_encode_dict['toss_winner'][toss_winner],'2','0']
inp = np.array(inp).reshape((1, -1))
print(inp)
output=model.predict(inp)
print(f"The winner would be: {list(team_encodings.keys())[list(team_encode_dict['team1'].values()).index(output)]}")