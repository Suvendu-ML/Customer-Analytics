# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy
import pickle
from werkzeug import secure_filename
import os
import datetime
import time
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
# EDA Packages
import pandas as pd
import numpy as np



app = Flask(__name__)
Bootstrap(app)
db = SQLAlchemy(app)
model = pickle.load(open('randomforestmodel.pkl', 'rb'))
# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'

# Saving Data To Database Storage
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)
	df_Pred = []

@app.route('/')
def index():
	return render_template('index.html')

# Route for our Processing and Details Page
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB',filename))
		fullfile = os.path.join('static/uploadsDB',filename)

		# For Time
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

		# EDA function
		acct = pd.read_csv(os.path.join('static/uploadsDB',filename))
		acct.drop(['PaperlessDate','invoiceNumber','InvoiceDate','DueDate','SettledDate'],axis = 1, inplace = True)
		# Categorical boolean mask
		categorical_feature_mask = acct.dtypes==object
		# filter categorical columns using mask and turn it into a list
		categorical_cols = acct.columns[categorical_feature_mask].tolist()
#		categorical_feature_not_mask = acct.dtypes!=object
#		# filter categorical columns using mask and turn it into a list
#		not_categorical_cols = acct.columns[categorical_feature_not_mask].tolist()
		le = LabelEncoder()
		# apply le on categorical feature columns
		acct[categorical_cols] = acct[categorical_cols].apply(lambda col: le.fit_transform(col))
		X = acct.iloc[:,0:6].values
#		y = acct.iloc[:,-1].values
		scaler = StandardScaler()
		X = np.array(X)
		X = scaler.fit_transform(X)
		X = pd.DataFrame(X)

		   
		df_Pred = model.predict(X)
		X['Pred'] = df_Pred
		
		for i in range(0,len(df_Pred)):
			if df_Pred[i] == 0:
				df_Pred[i] = 0
			elif (df_Pred[i] > 0) and (df_Pred[i] <= 15):
				df_Pred[i] = 1
			elif (df_Pred[i] > 15) and (df_Pred[i] <= 30):
				df_Pred[i] = 2
			elif (df_Pred[i] > 30) and (df_Pred[i] <= 90):
				df_Pred[i] = 3
				
		acct['Pred'] = df_Pred
		gr_val = acct.groupby('Pred')['InvoiceAmount'].sum()	
 			
		data_index2 = []
		data_index3 = [] 
	     
		df_Pred = pd.Index(df_Pred)
		data = df_Pred.value_counts()
		data[len(data)] = 0
		data_index = data.index
#		for i in range (0,len(data_index)):
#			if data_index[i] == 0:
#				data_index2 = 'On Time'
#			elif data_index[i] == 1:
#				data_index2 = '0-15'
#			elif data_index[i] == 2:
#				data_index2 = '15-30'
#			elif data_index[i] == 3:
#				data_index2 = '30-90'
#			elif data_index[i] == 4:
#				data_index2 = 'null'
				
		data_index2 = np.array(data_index2)	
		data_values = data.values
		total = data.values.sum()
		data_values_per = (data/total)*100
		data_values_per_index = data_values_per.index
#		for i in range (0,len(data_values_per_index)):
#			if data_index[i] == 0:
#				data_index3 = 'On Time in % '
#			elif data_index[i] == 1:
#				data_index3 = '15 Days delay in %'
#			elif data_index[i] == 2:
#				data_index3 = '15 Days to 30 Days delay in %'
#			elif data_index[i] == 3:
#				data_index3 = 'More than 30 days delay in %'
				
				
		labels = [ 'On Time', '1-15 days', '16-30 days', '31-90 days']
				
	   
		db.session.commit()
			
		return render_template('chart.html', title='Customer Payment Delay Prediction', max=17000, 
						 labels=labels, values=data_values,
						 per_labels = labels, 
						 per_values = data_values_per,
						 invoice_value = gr_val.values )            
#         
            
if __name__ == '__main__':
	app.run(debug = True)





