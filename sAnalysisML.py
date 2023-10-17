'''
 First we'll import our libraries

'''

import pandas as pd 
import matplotlib.pyplot as plt         
import seaborn as sns 
from wordcloud import WordCloud #visualizing text data in the form of clouds 
import re  #handling strings/regular expressions 



#for NLP 
import nltk #collection of libraries for nlp 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer


#for ML stuff 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #for transforming text into vectors 
from sklearn.model_selection import GridSearchCV #for hyperparam tuning of ML models 
from sklearn.ensemble import RandomForestClassifier #algo for classif


#metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, classification_report


#read the data 
df = pd.read_csv('Datasets/training.1600000.processed.noemoticon.csv',
                 delimiter=',', encoding='ISO-8859-1')
df.columns = ['Sentiment','id','date','query','user','text']
df = df[['Sentiment','text']]


#analyze our df 
print("Shape of the df : ", df.shape)
#df.sample(5)

#check our label values (interesting to see what are the values of our sent column) 
#sns.countplot(df.Sentiment)



#perform lemmatization (converting each word into its lexeme )

#create an object of WordNetLemmatizer
lm = WordNetLemmatizer()

def text_transformation(df_col):
    corpus = []
    for item in df_col : 
        #get rid of any chars apart from the alphabet
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        #convert to lowercase 
        new_item = new_item.lower()
        new_item = new_item.split()
        #get rid of stop words and perform lemmatization 
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        #add to corpus
        corpus.append(' '.join(str(x) for x in new_item))
    
    return corpus 


#apply to our data 
#return a corpus of processed data 
corpus = text_transformation(df['text'])



'''
Then we will use the Bag of Words Model (BOW)-> 
    represent the text in the form of a bag of words , multiplicity here is the main point of concern
'''
#we can use Scikit-Learn to do that using CountVectorizer
#convert our text data into vectors ,by fitting and transforming the corpus we just created 

cv = CountVectorizer(ngram_range= (1,2))
traindata = cv.fit_transform(corpus)
X = traindata 
y = df.Sentiment


#ngram -> sequence of 'n' of words in a row or sentence
#ngram_range -> parameter we use to give importance to the combination of words 


#we ll also use GridSearch cross validation to fit our estimators on the training data with 
 # all possible combinations of predefined params, which we will feed 


#create a dictionnary of parameters
parameters = {'max_features': ('auto', 'sqrt'),
              'n_estimators' : [500, 1000, 1500],
              'max_depth': [5, 10, None],
              'min_samples_split' : [5, 10, 15],
              'min_samples_leaf' : [1, 2, 5, 10],
              'bootstrap' : [True, False] }


#feed the params to GridSearchCV 
grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
grid_search.fit(X,y)
grid_search.best_params_ 


#view all the models and their respective parameters 
for i in range(432):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test Score : '
    ,grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])


#now choose the best params obtained from GridSearchCV and create our rfc model
rfc = RandomForestClassifier(max_features=grid_search.best_params_['max_features'],
                                      max_depth=grid_search.best_params_['max_depth'],
                                      n_estimators=grid_search.best_params_['n_estimators'],
                                      min_samples_split=grid_search.best_params_['min_samples_split'],
                                      min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                      bootstrap=grid_search.best_params_['bootstrap'])





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rfc.fit(X_train, y_train)

# Make predictions
y_pred = rfc.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification)




