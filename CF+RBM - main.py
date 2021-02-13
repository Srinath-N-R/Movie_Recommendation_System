import tensorflow as tf
import pandas as pd

movies_df = pd.read_csv('/resources/data/ml-1m/movies.dat', sep='::', header=None)

ratings_df = pd.read_csv('/resources/data/ml-1m/ratings.dat', sep='::', header=None)

movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

movies_df['List Index'] = movies_df.index

merged_df = movies_df.merge(ratings_df, on='MovieID')
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

userGroup = merged_df.groupby('UserID')
userGroup.first().head()


amountOfUsedUsers = 1000
trX = []
for userID, curUser in userGroup:
    temp = [0]*len(movies_df)
    for num, movie in curUser.iterrows():
        temp[movie['List Index']] = movie['Rating']/5.0
    trX.append(temp)
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1

user_id = int(input('Enter User ID: '))

inputUser = [trX[user_id]]

a = pd.read_csv('a.csv', header=False)
[v0, W, hb, vb, prv_w, prv_hb, prv_vb] = a.iloc[0, :]
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = tf.sess.run(hh0, feed_dict={ v0: inputUser, W: prv_w, hb: prv_hb})
rec = tf.sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb})

movies_df["Recommendation Score"] = rec[0]

print('Your recommendations are: ')
print(movies_df.sort(["Recommendation Score"], ascending=False).head(20))