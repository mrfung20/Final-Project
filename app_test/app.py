import flask
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')

soup_df = pd.read_csv('../Resources/mini_meta_soup.csv')

count_vector = CountVectorizer(stop_words='english')
count_matrix = count_vector.fit_transform(soup_df['soup'])

cosine_score = cosine_similarity(count_matrix, count_matrix)

soup_df = soup_df.reset_index()
index_df = pd.Series(soup_df.index, index=soup_df['title']).drop_duplicates()
all_titles = [soup_df['title'][i] for i in range(len(soup_df['title']))]

def get_recommendations(title, cosine_score=cosine_score):
    # retrieve movie index using title
    idx = index_df[title]
    # list of similarity scores - tuples contain index and cosine similarity
    scores_list = list(enumerate(cosine_score[idx]))
    # sort list in descending order by similarity score
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    # grab similarity score for top 5 recommendations
    scores_list = scores_list[1:6]
    # grab index for top 5 recommendations
    movie_index = [i[0] for i in scores_list]
    # map index to respective title and return recommendations
    movie_recs = soup_df['title'].iloc[movie_index]
    return movie_recs

# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()

        if m_name not in all_titles:
            return(flask.render_template('negative.html',name=m_name))
        else:
            result_final = get_recommendations(m_name)
            names = []
#            dates = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
#                dates.append(result_final.iloc[i][1])

            return flask.render_template('positive.html',movie_names=names,movie_date=dates,search_name=m_name)

if __name__ == '__main__':
    app.run()
