import pickle
import streamlit as st
model=pickle.load(open('multi_class_classification_model.pkl','rb'))

def mymodel():
    st.title("Multi-class Classification Model")
    a1=st.number_input('Enter sl')
    b2=st.number_input('Enter sw')
    c3=st.number_input('Enter pl')
    d4=st.number_input('Enter pw')

    pred=st.button('predict')

    if pred:
        result=model.predict([[a1,b2,c3,d4]])
        st.success(f'The predicted class is: {result[0]}')  
    
        
        
mymodel()


