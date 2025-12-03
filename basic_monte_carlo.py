import numpy as np
import matplotlib.pyplot as plt

def run_portfolio_simulation(capital, expected_return, volatility, days, simulations):
    dt = 1 / days
    pmat = np.zeros((days, simulations))
    pmat[0] = capital
    for t in range(1, days):
        # Creating standard normal distribution (Z score for each day)
        Z = np.random.normal(0, 1, simulations)

        # Calculating the Drift
        # Formula: (mu - 0.5 * sigma^2) * dt
        drift = (expected_return - 0.5 * volatility**2 ) * dt
        
        # Calculating the Diffusion/Shock (Random part)
        # Formula: sigma * sqrt(dt) * Z
        diffusion = volatility * np.sqrt(dt) * Z
        pmat[t] = pmat[t-1] * np.exp(drift + diffusion)
        
    return pmat

if __name__ == "__main__":
    starting_money = 80000
    expected_return = 0.15  # (savings 5%, Bonds 6%, S&P 500 10%, Tech stocks 15%, crypto 20%)
    vol = 0.50    # (higher expected return, higher risk volatility should be, based on how volatile the stocks invested are)
    time_span = 252 # 252 trading days in a year
    num_sims = 100     # 100 simulations

    results = run_portfolio_simulation(starting_money, expected_return, vol, time_span, num_sims)
    
    plt.figure(figsize=(15,9))
    plt.plot(results)
    plt.title("Monte Carlo Portfolio Simulation by Robert")
    plt.xlabel("Days since starting")
    plt.ylabel("Portfolio Value ($)")
    plt.show()
    
    final_values = results[-1]
    print(f"Expected Mean Value: ${np.mean(final_values):,.2f}")
    print(f"Worst Case (5th Percentile): ${np.percentile(final_values, 5):,.2f}")
