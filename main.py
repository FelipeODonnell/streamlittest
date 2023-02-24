'''
****
Create a virtual environment

Used to create env:
python3 -m venv DLpricer

Activate:
source DLpricer/bin/activate

Deactive:
deactivate


***

To run
streamlit run main.py

Types of learning models to try
Multi-layer perceptron (MLP)
Convolutional neural network (CNN) 
Recurrent neural network (RNN)
Long short-term memory (LSTM)
'''


#Import 
from operator import index
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.stats import norm
import altair as alt


with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Blueblocks.ai")
    choice = st.radio("Navigation", ["Valuation","Structuring","Swaps", "Market"])
    st.info("This project application helps you get instant valuations and structures solutions for your property")

if choice == "Valuation":
    pass



if choice == "Structuring": 
        # Load historical UK property data
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv(data_url, header=None, delimiter=r"\s+", names=columns)

    # Define function to calculate future prices
    def calc_future_prices(purchase_price, rental_price):
        # Calculate annual growth rates from historical data
        purchase_growth = np.mean(data['MEDV'].pct_change())
        rental_growth = np.mean(data['RM'].pct_change())
        
        # Calculate future prices for every year for the next 7 years
        future_purchase_prices = []
        future_rental_prices = []
        for year in range(1, 8):
            future_purchase_price = purchase_price * (1 + purchase_growth)**year
            future_rental_price = rental_price * (1 + rental_growth)**year
            future_purchase_prices.append(future_purchase_price)
            future_rental_prices.append(future_rental_price)
        
        return future_purchase_prices, future_rental_prices

    # Define Streamlit app
    st.title('Future Property Prices')
    st.write('Enter the current sale and rental price of your property in the UK:')
    purchase_price = st.number_input('Sale Price', value=0)
    rental_price = st.number_input('Rental Price', value=0)


    if st.button('Calculate Future Prices'):
        # Calculate future prices
        future_purchase_price, future_rental_price = calc_future_prices(purchase_price, rental_price)
        
        # Display future prices in a table
        future_prices_df = pd.DataFrame({'Year': range(1, 8), 
                                        'Future Purchase Price': future_purchase_price, 
                                        'Future Rental Price': future_rental_price})
        st.write(future_prices_df)
        
        # Plot future price vs year
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 8), future_purchase_price)
        ax1.set_title('Future Purchase Price vs Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Future Purchase Price (£)')
        st.pyplot(fig1)
        
        # Plot future rental vs price
        fig2, ax2 = plt.subplots()
        ax2.plot(future_purchase_price, future_rental_price)
        ax2.set_title('Future Rental Price vs Future Purchase Price')
        ax2.set_xlabel('Future Purchase Price (£)')
        ax2.set_ylabel('Future Rental Price (£)')
        st.pyplot(fig2)


if choice == "Swaps": 
    def calculate_fixed_payments(current_payment, outstanding_amount):
    # Calculate the monthly fixed mortgage payment for the next year with a 10% fee
        fixed_payment = (current_payment * 1.15) * 12
        return fixed_payment

    def calculate_cash_flows(fixed_payment, outstanding_amount):
        # Calculate the monthly cash flows for the fixed payments
        cash_flows = []
        balance = outstanding_amount
        for i in range(12):
            interest = balance * 0.05 / 12
            principal = fixed_payment - interest
            balance -= principal
            cash_flows.append(fixed_payment)
        return cash_flows

    st.title("Mortgage Payment Calculator")

    current_payment = st.number_input("Enter your current monthly variable mortgage payment:", min_value=0.01, step=0.01)
    outstanding_amount = st.number_input("Enter your outstanding mortgage amount:", min_value=0.01, step=0.01)

    if st.button("Calculate"):
        fixed_payment = calculate_fixed_payments(current_payment, outstanding_amount)
        st.write("Your monthly fixed mortgage payment per month would be:", fixed_payment/12)
        cash_flows = calculate_cash_flows(fixed_payment/12, outstanding_amount)
        cash_flows_df = pd.DataFrame({"Month": range(1, 13), "Payment": cash_flows})
        st.write("Monthly cash flows for the fixed payments:")
        st.dataframe(cash_flows_df)
            

if choice == "Market": 
    st.title("Example of Derivatives Securities Market")
    st.image('333368727_567698838713526_2956667945644696937_n.jpg')
