'''
****
Create a virtual environment

Used to create env:
python3 -m venv streamlittestv1 

Activate:
source streamlittestv1/bin/activate

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
    choice = st.radio("Navigation", ["Valuation","Structuring","Swaps"])
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
    
    # Define function to calculate swap price
    def calc_swap_price(change_fixed, value, time_period, notional, fixed_rate, variable_rate, volatility):
        
        # Calculate present value of fixed and variable cash flows
        fixed_present_value = fixed_rate * notional * (1 - 1 / ((1 + fixed_rate)**time_period)) / (fixed_rate)
        variable_present_value = notional * (1 - 1 / ((1 + variable_rate)**time_period))
        
        # Calculate swap price
        if change_fixed:
            # Change fixed to variable
            d1 = (np.log(variable_rate / fixed_rate) + 0.5 * volatility**2 * time_period) / (volatility * np.sqrt(time_period))
            d2 = d1 - volatility * np.sqrt(time_period)
            swap_price = (fixed_present_value - variable_present_value) * norm.cdf(-d2) - value * norm.cdf(-d1)
        else:
            # Change variable to fixed
            d1 = (np.log(variable_rate / fixed_rate) - 0.5 * volatility**2 * time_period) / (volatility * np.sqrt(time_period))
            d2 = d1 - volatility * np.sqrt(time_period)
            swap_price = value * norm.cdf(d1) - (fixed_present_value - variable_present_value) * norm.cdf(d2)
            
        return swap_price

    # Define Streamlit app
    st.title('Mortgage Rate Swapper')
    st.write('Swap Pricer - Select Fixed to Variable or Variable to Fixed:')
    change_fixed = st.selectbox('Change Fixed to Variable or Variable to Fixed?', ['Fixed to Variable', 'Variable to Fixed'])
    value = st.number_input('Value of Swap', value=0.0)
    time_period = st.number_input('Time Period (in years)', value=0.0)
    notional = st.number_input('Notional Value', value=1000000.0)
    fixed_rate = st.number_input('Fixed Rate', value=0.05)
    variable_rate = st.number_input('Variable Rate', value=0.04)
    volatility = st.number_input('Volatility', value=0.2)

    if st.button('Calculate Swap Price'):
        # Calculate swap price
        swap_price = calc_swap_price(change_fixed=='Fixed to Variable', value, time_period, notional, fixed_rate, variable_rate, volatility)
        
        # Display swap price
        st.write('The price of the swap is:', swap_price)
        
        # Calculate monthly payments
        fixed_payment = fixed_rate * notional / 12
        variable_payment = variable_rate * notional / 12
        swap_payment = swap_price / (time_period * 12)
        data = {'Month': [], 'Fixed Payment': [], 'Variable Payment': [], 'Swap Payment': []}
        for month in range(int(time_period * 12)):
            data['Month'].append(month + 1)
            data['Fixed Payment'].append(fixed_payment)
            data['Variable Payment'].append(variable_payment)
            data['Swap Payment'].append(swap_payment)
        payments = pd.DataFrame(data)
        
        # Display monthly payments table
        st.write('Monthly Payments:')
        st.dataframe(payments)
        
        # Display future value vs year chart
        data = {'Year': [], 'Future Value': [], 'Future Rental': []}
        for year in range(1, 8):
            future_value = notional * (1 + fixed_rate)**year
        


