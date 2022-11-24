import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from PIL import Image
import pickle
import Orange
# load model 
import joblib
from sklearn.svm import SVC
model=pickle.load(open('./svcapk.sav','rb'),buffers=None)
# linear programming
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

def main():
    """App with Streamlit"""
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(layout="wide")
    st.title("PREDIKSI KUALITAS LAPORAN KEUANGAN KEMENTERIAN LEMBAGA")
    menu = ['Pilih Analisis', 'Visualisasi Data','Prediksi Kualitas LK']
    menu1 = ['Pilih Prediksi', 'Kualitas LKKL', 'Kualitas LKPD', 'Kualitas LK-BUN']    
    pjjda= Image.open('./APK.jpg')
    st.sidebar.image(pjjda)
    
    C1, C2, C3 = st.columns((2,1,2))
    
    choice = st.sidebar.selectbox("Select Menu", menu)
    parameter_select = st.sidebar.selectbox("Pilih Jenis Prediksi", menu1)
        
    if choice == "Prediksi Kualitas LK" and parameter_select != "Pilih Prediksi" :
        #st.subheader("Prediksi Pendapatan")
        
        with C1:
            st.subheader("Input Variabel Predictor")
            a = st.number_input(label="Ast",value=15330669000000.00,min_value=0.0, max_value=100000000000000.0, step=0.1)
            b = st.number_input(label="Real",value=8300000000000.00,min_value=0.0, max_value=100000000000000.0, step=0.1)
            c = st.number_input(label="DIPA",value=100,min_value=1, max_value=10000, step=1)
            d = st.number_input(label="PgMin",value=10.00,min_value=0.0, max_value=100.0, step=0.1)
            e = st.number_input(label="Kont",value=10.00,min_value=0.0, max_value=100.0, step=0.1)
            f = st.number_input(label="UPTUP",value=10.00,min_value=0.0, max_value=100.0, step=0.1)
            g = st.number_input(label="LPJ",value=10.00,min_value=0.0, max_value=100.0, step=0.1)
            h = st.number_input(label="TAG",value=10.00,min_value=0.0, max_value=100.0, step=0.1)
            i = st.number_input(label="SPM",value=10.00,min_value=0.0, max_value=100.0, step=0.1)
            j = st.number_input(label="PolNonPol",value=1,min_value=0, max_value=1, step=1)
            k = st.number_input(label="TkPdkPim",value=0,min_value=0, max_value=3, step=1)
            l = st.number_input(label="JnPdkPim",value=1,min_value=0, max_value=1, step=1)
            m = st.number_input(label="Umur",value=32,min_value=1, max_value=100, step=1)
            n = st.number_input(label="FraudKas",value=8300000000000.00,min_value=0.0, max_value=100000000000000.0, step=0.1)
            o = st.number_input(label="FraudNon",value=8300000000000.00,min_value=0.0, max_value=100000000000000.0, step=0.1)
            p = st.number_input(label="Temuan_gbg",value=26,min_value=1, max_value=100, step=1)
            q = st.number_input(label="Tinjut_gbg",value=11.00,min_value=0.0, max_value=100.0, step=0.1)
            r = st.number_input(label="penyerapan",value=87.00,min_value=0.0, max_value=100.0, step=0.1)
            
        with C2:
            st.write = ""
        
        with C3:
            st.subheader("Prediksi Kualitas LKKL  \nBerdasarkan Machine-Learning Models")                
            if st.button("Klik untuk Mendapatkan Prediksi"):
                dfvalues = pd.DataFrame(list(zip([a],[b],[c],[d],[e],[f],[g],[h],[i],[j],[k],[l],[m],[n],[o],[p],[q],[r])),
                columns =['Ast','Real','DIPA','PgMin','Kont','UPTUP','LPJ','TAG','SPM','PolNonPol',
                'TkPdkPim','JnPdkPim','Umur','FraudKas','FraudNon','Temuan_gbg','Tinjut_gbg','penyerapan'])
                input_variables = np.array(dfvalues[['Ast','Real','DIPA','PgMin','Kont','UPTUP','LPJ','TAG','SPM','PolNonPol',
                'TkPdkPim','JnPdkPim','Umur','FraudKas','FraudNon','Temuan_gbg','Tinjut_gbg','penyerapan']])
                predict =model.predict(dfvalues)
                # st.dataframe(predict)
                # st.text(predict[0])
                if predict[0] == 2:
                    st.header(' WTP')
                elif predict[0] == 1:
                    st.header(" WDP")
                else :
                    st.text(" TMP")
            #st.write(parameter_select)

                if parameter_select == "pph":
                    model_pph= open("pph_model.pkl", "rb")
                    pph=joblib.load(model_pph)
                
                    prediction_pph = pph.predict(input_variables)
                    nilai_pph = np.mean(prediction_pph)
                    nilai_pph = (np.exp(nilai_pph))/1000000000000
                    st.title(f"Prediksi Nilai PPH: Rp{nilai_pph:.2f} Trilliun")

                elif parameter_select == "PPN":
                    model_ppn= open("ppn_model.pkl", "rb")
                    ppn=joblib.load(model_ppn)
                
                    prediction_ppn = ppn.predict(input_variables)
                    nilai_ppn = np.mean(prediction_ppn)
                    nilai_ppn = (np.exp(nilai_ppn))/1000000000000
                    st.title(f"Prediksi Nilai PPN: Rp{nilai_ppn:.2f} Trilliun")

                elif parameter_select == "PBB":
                    model_pbb= open("pbb_model.pkl", "rb")
                    pbb=joblib.load(model_pbb)
                
                    prediction_pbb = pbb.predict(input_variables)
                    nilai_pbb = np.mean(prediction_pbb)
                    nilai_pbb = (np.exp(nilai_pbb))/1000000000
                    st.title(f"Prediksi Nilai PBB: Rp{nilai_pbb:.2f} Milyar")

                elif parameter_select == "CUKAI":
                    model_cukai= open("cukai_model.pkl", "rb")
                    cukai=joblib.load(model_cukai)
                
                    prediction_cukai = cukai.predict(input_variables)
                    nilai_cukai = np.mean(prediction_cukai)
                    nilai_cukai = (np.exp(nilai_cukai))/1000000000
                    st.title(f"Prediksi Nilai CUKAI: Rp{nilai_cukai:.2f} Milyar")

                elif parameter_select == "PAJAK LAIN":
                    model_pjklain= open("pjklain_model.pkl", "rb")
                    pjklain=joblib.load(model_pjklain)
                
                    prediction_pjklain = pjklain.predict(input_variables)
                    nilai_pjklain = np.mean(prediction_pjklain)
                    nilai_pjklain = (np.exp(nilai_pjklain))/1000000000
                    st.title(f"Prediksi Nilai PAJAK LAIN: Rp{nilai_pjklain:.2f} Milyar")

                elif parameter_select == "BEA MASUK":
                    model_beamas= open("beamas_model.pkl", "rb")
                    beamas=joblib.load(model_beamas)
                
                    prediction_beamas = beamas.predict(input_variables)
                    nilai_beamas = np.mean(prediction_beamas)
                    nilai_beamas = (np.exp(nilai_beamas))/1000000000
                    st.title(f"Prediksi Nilai BEA MASUK: Rp{nilai_beamas:.2f} Milyar")

                elif parameter_select == "BEA KELUAR":
                    model_beakel= open("beakel_model.pkl", "rb")
                    beakel=joblib.load(model_beakel)
                
                    prediction_beakel = beakel.predict(input_variables)
                    nilai_beakel = np.mean(prediction_beakel)
                    nilai_beakel = (np.exp(nilai_beakel))/1000000000
                    st.title(f"Prediksi Nilai BEA KELUAR: Rp{nilai_beakel:.2f} Milyar")
                
    elif choice == "Basis Akrual" :
        with C1:
            title = choice+' masih dalam pengembangan'
            st.subheader(title)

    elif choice == "Pilih Legder":
        with C1:
            title = 'Silahkan pilih terlebih dahulu ledger pada sidebar disamping'
            st.subheader(title)
        
    elif parameter_select == "Pilih Jenis Prediksi":
        with C1:
            title = 'Silahkan pilih terlebih dahulu Jenis Pendapatan pada sidebar disamping'
            st.subheader(title)

if __name__=='__main__':
    main()
