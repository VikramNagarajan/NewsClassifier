#importing modules
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from webdriver_manager.chrome import ChromeDriverManager
import tkinter as tk
import tkinter.font as tkFont

#opening file
data = pd.read_csv('news.tsv', sep='\t', na_filter=False)

#transforming tweets into matrices
tf = TfidfVectorizer()
text_tf = tf.fit_transform(data['url'])

#splitting train file train and test
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['primary.topic'], test_size=0.10, random_state=11)

#ML Model
clf = SGDClassifier().fit(X_train, y_train)

#GUI w/Tkinter
root = tk.Tk()
myFont = tkFont.Font(family='Courier', size=20)
myFont2 = tkFont.Font(family='Courier', size=40)
root.title('Is this racist')
article = tk.Entry()
article.insert('0', 'URL')
x = tk.Label(text='', font=myFont2)

predict2 = clf.predict(X_test)
print(metrics.accuracy_score(y_test, predict2))

#button command
def ml():
    predict = clf.predict(tf.transform([article.get()]))
    x.configure(text='Your topic is: ' + str(predict[0]))
    

#packing widgets 
filler = tk.Label(text='')
filler2 = tk.Label(text='')
submit = tk.Button(text='Submit', bg='forestGreen', fg='white', command=ml)
filler2.pack()
article.pack()
filler.pack()
submit.pack()
x.pack()
print()
root.mainloop()