
import pickle
import streamlit as st

model1=pickle.load(open('areaprice.pkl','rb'))
def mydeploy():
    st.title('Area price prediction')
    area=st.number_input('Enter area in square feet')
   
    pred=st.button('Predict')
    if pred:
        x=model1.predict([[area]])
        st.write('The predicted price is:', x[0])

mydeploy()

                        

        