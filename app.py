
import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('reg_model.pkl', 'rb'))

@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')


@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':

        # getting the feature inputs from the user 
        
        Medu = request.form.get("Medu")	
        Fedu = request.form.get("Fedu")
        health = request.form.get("Health")
        absences = request.form.get("absences")
        G1	= request.form.get("G1")
        G2 = request.form.get("G2")

        # arranging the inputs in an array 
        input_list = [Medu,Fedu,health,absences,G1,G2]
        array = np.array(input_list)

        # prediction using model.predict 
        output = model.predict(array.reshape(1,-1))
        output =  round(output[0],2)


        # return the page with updated predicted text 
        return render_template('index.html', prediction_text= "G3 marks is {}".format(output))
    else:
        return render_template('index.html')

    
if __name__ == "__main__":
    app.run(debug=True)
    


# added comment 





