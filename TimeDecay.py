# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 21:59:39 2023

@author: sigma
"""
import mibian as mb

# Input parameters for the Black-Scholes model
spot_price = 15500
strike_price = 15000
interest_rate = 0.0145
dividend_yield = 0
volatility = 0.30
time_to_expiration = 90
initial_time_to_expiration = 90 # time to expiration in days
end_time_to_expiration=3

bs_model=mb.BS([spot_price, strike_price, interest_rate, initial_time_to_expiration, dividend_yield], volatility=volatility)

initial_bs_model = mb.BS([spot_price, strike_price, interest_rate, initial_time_to_expiration, dividend_yield], volatility=volatility)

end_bs_bs_model = mb.BS([spot_price, strike_price, interest_rate, end_time_to_expiration, dividend_yield], volatility=volatility)

# Calculate the initial price and time value of the call option



# Calculate the initial theta for the put option
initial_theta = (initial_bs_model.callTheta)*50
initial_theta 

end_theta=(end_bs_bs_model.callTheta)*50
end_theta

# Simulate the theta decay over time for the put option
for days_left in range(time_to_expiration, 0, -1):
    # Calculate the new theta of the put option
    bs_model = mb.BS([spot_price, strike_price, interest_rate, days_left, dividend_yield], volatility=volatility)
    initial_bs_model = mb.BS([spot_price, strike_price, interest_rate, initial_time_to_expiration/365, dividend_yield], volatility=volatility)
    end_bs_bs_model = mb.BS([spot_price, strike_price, interest_rate, end_time_to_expiration/365, dividend_yield], volatility=volatility)
    initial_theta = initial_bs_model.callTheta
    end_theta=end_bs_bs_model.callTheta
    
    # Calculate the theta decay
    theta_decay = initial_theta - end_theta
    
    # Print the results
    print(f"Days left: {days_left}, Theta decay: {theta_decay:.2f}")