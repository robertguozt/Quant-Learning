import numpy as np
from scipy.stats import norm

#C = S0 N(d1) - K e^(-rT)N(d2)
#d1 = ln(S0/K)+(r+(sigma^2)/2)T / (sigma * sqrt(T))
#d2 = d1 - sigma * sqrt(T)

#N = normal cdf
#S = Stock price: current price of the stock
#K = Strike Price: The  price to buy or sell
#T = Time: Time to expiration in years
#r = Risk-Free Rate: The yield on a "safe" investment like a 10-year Treasury bond (e.g., 0.04 for 4%).
#sigma = volatility: standard deviation of the stocks returns (how much it swings)

#Greeks (aka the controls):
#DELTA = N(d1) (if delta = 0.6, and stock goes up $1, then option price goes up $0.6)
#GAMMA = N'(d1) / (S * sigma * sqrt(T))   (Acceleration of delta, where if gamma value is high, then delta change quickly, this is why stocks become explosive near expiration)
#VEGA = S * sqrt(T) * N(d1) (if vega is 0.15, and the market gets more volatile (volatility +1%), then option increases by $0.15 even though market stays still)
#Theta = time decay

def black_scholes_numpy(S, K, T, r, sigma, option_type="call"):
    #calculating d1 and d2
    d1 = (np.log(S / K)+(r + 0.5*(sigma**2)) * T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def black_scholes_extended(S, K, T, r, sigma, option_type="call"):
    # Calculate d1, d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Common derivative component used in Gamma and Vega
    pdf_d1 = norm.pdf(d1)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        # Theta for a call
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        # Theta for a put
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))

    # Gamma and Vega are the same for Calls and Puts
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1
    
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100, # Reported as price change per 1% change in vol
        "theta": theta / 365 # Reported as price change per day
    }
def main():
    S_range = np.linspace(50, 150, 100)
    # Calculate the call price for every stock price in that range at once
    call_prices = black_scholes_numpy(S_range, K=100, T=1, r=0.05, sigma=0.2)

    print(f"Calculated {len(call_prices)} prices instantly!")
def main_extended():
    result = black_scholes_extended(S=100, K=100, T=0.5, r=0.05, sigma=0.2)

    print(f"Price: {result['price']:.2f}")
    print(f"Delta: {result['delta']:.4f}")
    print(f"Daily Decay (Theta): {result['theta']:.4f}")

main()
main_extended()


