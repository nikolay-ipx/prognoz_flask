from flask import Flask,render_template, request, redirect
import pandas as pd
from flask import send_file
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)#,static_folder='flask')#название основного файла app.py


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/filedownload')
def get():
    return render_template('filedownload.html')


@app.route('/getfile',methods=['GET','POST'])
def getfile():
    if request.method == 'POST':
        file = request.files['myfile']
        prognoz(file)
        return redirect('/download')
    else:
        return render_template('filedownload.html')


@app.route('/download')
def download():
    return send_file('f_tt.csv')


def prognoz(fff):
    f=pd.read_csv(fff,sep=';')
    f=f.drop(columns=['POLICY_BRANCH','VEHICLE_MAKE','VEHICLE_MODEL','CLIENT_REGISTRATION_REGION',
                      'INSURER_GENDER','POLICY_BEGIN_MONTH','POLICY_END_MONTH','POLICY_SALES_CHANNEL',
                      'POLICY_SALES_CHANNEL_GROUP',
                      'VEHICLE_IN_CREDIT','POLICY_INTERMEDIARY','VEHICLE_SUM_INSURED',

                      'POLICY_HAS_COMPLAINTS','POLICY_DEDUCT_VALUE'
                      ])


    f=f.drop(f[f.POLICY_CLM_N == "n/d"].index)
    f=pd.get_dummies(f,columns=['POLICY_CLM_N'])
    f=pd.get_dummies(f,columns=['POLICY_CLM_GLT_N'])
    f=pd.get_dummies(f,columns=['POLICY_PRV_CLM_N'])
    f=pd.get_dummies(f,columns=['POLICY_PRV_CLM_GLT_N'])


    f_o=f.drop(f[f.DATA_TYPE == "TEST "].index)
    f_o=f_o.drop(columns=['DATA_TYPE','POLICY_ID'])


    f_t=f.drop(f[f.DATA_TYPE == "TRAIN"].index)
    f_POLICY_ID=f_t['POLICY_ID']
    f_t=f_t.drop(columns=['DATA_TYPE','POLICY_ID'])


    X=f_o
    X=X.drop(columns=['POLICY_IS_RENEWED'])
    y=f_o['POLICY_IS_RENEWED']


    model_tree = DecisionTreeClassifier(max_depth=6)
    model_tree.fit(X,y)
    f_t=f_t.drop(columns=['POLICY_IS_RENEWED'])
    predict=model_tree.predict(f_t)


    f_tt=pd.DataFrame({'POLICY_ID':f_POLICY_ID,'POLICY_IS_RENEWED':predict})
    f_tt.to_csv('f_tt.csv',index=False)
    print(f_tt)


if __name__ == '__main__':
    app.run(debug=True)