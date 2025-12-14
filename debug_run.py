from lcp.simulation import SimulationRunner
import pandas as pd
import time

def test_sim():
    print("Initializing Runner...")
    runner = SimulationRunner("Koster direct normal irradiance_Wh per square meter.csv")
    
    print("Loading Data...")
    df_in = runner.load_data()
    print("Input Data Head:")
    print(df_in.head())
    
    print("\nRunning Year Simulation (Dry Run 100 steps)...")
    # Hack to run short loop
    # We will just call the main run but interrupt or just run full?
    # It's fast (matrix math). 365*24*10 = 87600 steps.
    # Should take < 10 seconds.
    
    start = time.time()
    res = runner.run_year()
    end = time.time()
    
    print(f"\nSimulation Complete in {end-start:.2f}s")
    print(f"Result Shape: {res.shape}")
    print(res.head())
    print("\nTotal Energy (kWh):")
    total_wh = res['Actual_Power'].sum() * (6.0/60.0) # Power (W) * Hours
    print(f"{total_wh/1000.0:.2f} kWh")
    
    # Check for Safety Mode triggers
    safeties = res[res['Safety_Mode'] == True]
    print(f"\nSteps with Safety Mode Triggered: {len(safeties)}")
    if len(safeties) > 0:
        print(safeties.head())

if __name__ == "__main__":
    test_sim()
