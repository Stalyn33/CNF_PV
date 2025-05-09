import pandas as pd
import numpy as np
import sys, os, torch, json, joblib
import torch.nn.functional as F
from train_sample import generate_synthetic_data
from sklearn.metrics import mean_absolute_error
from fastdtw import fastdtw
from plot_samples import plot_samples_by_month, plot_samples_all_months
from maf_model_2 import MAFModel_2
from train_sample import plot_training_loss

def load_data(file_path, delimiter=','):
    """Load the CSV dataset."""
    return pd.read_csv(file_path, delimiter=delimiter)

def calculate_representative_curves(data):
    """Calculate the average daily profile for each month."""
    representative_curves = data.groupby('month').mean()
    representative_curves.reset_index(inplace=True)
    return representative_curves

def calculate_metrics(real_curves, synthetic_curves):
    """Calculate metrics between real and synthetic curves."""
    metrics = []
    months = real_curves['month']
    
    for month in months:
        real_values = real_curves[real_curves['month'] == month].iloc[:, 1:].values.flatten()
        synthetic_values = synthetic_curves[synthetic_curves['month'] == month].iloc[:, 1:].values.flatten()
        
        # MAE
        mae = mean_absolute_error(real_values, synthetic_values)
        
        # RMSE
        rmse = np.sqrt(np.mean((real_values - synthetic_values) ** 2))
        
        # Pearson Correlation Coefficient
        pcc = np.corrcoef(real_values, synthetic_values)[0, 1]
        
        # DTW (Dynamic Time Warping)
        distance, _ = fastdtw(real_values, synthetic_values)
        
        # Append metrics for the month
        metrics.append({
            "month": month,
            "MAE": mae,
            "RMSE": rmse,
            "PCC": pcc,
            "DTW": distance
        })
    
    return metrics

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#--------------------------------------- MAIN -------------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

def main():
    
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    #-------------------------------- LOAD THE MAF_MODEL ------------------------------------
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    # Verificatndo que existan los archivos necesarios para la evaluación
    required_files = ["maf_model.pt", "maf_model_config.json", "scaler.pkl"]

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print("The following required files are missing:")
        for f in missing:
            print(f" - {f}")
        print("Please run the training script before evaluating.")
        sys.exit()
    
    
    # Activación como aprámetros aceptables en JSON
    activation_map = {"relu": F.relu, "leaky_relu": F.leaky_relu, "elu": F.elu,
    "selu": F.selu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "softplus": F.softplus}
        
    #Load the scaler and the saved model parameters from the JSON file
    
    scaler = joblib.load("scaler.pkl")
    
    with open("maf_model_config.json", "r") as f:
        params = json.load(f)

    #-------------------------------------------------------------------------------------------------
    # Rebuild model with parameters (Choose betwen the simple "MAFModel" and editable model "MAFModel_2")
    
    maf_model = MAFModel_2(num_inputs=params["num_inputs"], condition_size=params["condition_size"],
    num_hidden=params["num_hidden"], num_layers=params["num_layers"],
    num_camadas=params["num_camadas"], residual=params["residual"], rndm_mask=params["rndm_mask"],
    fun_act=activation_map[params["fun_act"]], drop_prob=params["drop_prob"], norm_batch=params["norm_batch"])
    
    samples = 500         # Number of  sintetich samples to be generated per month
    #-------------------------------------------------------------------------------------------------
    # Load the model weights and biases
    maf_model.load_state_dict(torch.load("maf_model.pt"))
    maf_model.eval()
    
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    #--------------------------------- GENERATE SAMPLES -------------------------------------
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    csv_file_path = 'synthetic_data_profiles.csv'

    # Verify if the CSV file already exists
    if not os.path.exists(csv_file_path):
        generate_new = True
    else:
        # Ask to user for new samples
        while True:
            user_input = input(f"'{csv_file_path}' already exists. Do you want to generate new samples? (Y/N): ").strip().upper()
            if user_input in ['Y', 'N']:
                generate_new = (user_input == 'Y')
                break
            else:
                print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")

    if generate_new:
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(maf_model, scaler, num_samples=samples)

        # Convert to DataFrame (24 columns for profiles + 1 column for month)
        df = pd.DataFrame(synthetic_data)

        # Define the header row
        header = [f'Hour_{i}' for i in range(1, 25)] + ['month']

        # Assign the header
        df.columns = header

        # Save to CSV with the header
        df.to_csv(csv_file_path, index=False)

        print(f'New synthetic data profiles saved to {csv_file_path}')
    else:
        print(f'Using existing synthetic data in {csv_file_path}')
    
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    #------------------------------------- METRICS  -----------------------------------------
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    
    # Load the real dataset
    file_path = 'data_profiles_labeled.csv'  # Replace with the correct path
    data = load_data(file_path, delimiter=',')
    
    # Calculate the representative curves for the real dataset
    representative_curves = calculate_representative_curves(data)
    print("Representative Curves (Real Data):")
    print(representative_curves)
    
    # Load the synthetic dataset
    synthetic_file_path = 'synthetic_data_profiles.csv'  # Replace with the correct path
    synthetic_data = load_data(synthetic_file_path, delimiter=',')
    
    # Calculate the representative curves for the synthetic dataset
    synthetic_representative_curves = calculate_representative_curves(synthetic_data)
    print("Representative Curves (Synthetic Data):")
    print(synthetic_representative_curves)
    
    # Calculate metrics
    metrics = calculate_metrics(representative_curves, synthetic_representative_curves)
    
    # Convert the results to a DataFrame 
    metrics_df = pd.DataFrame(metrics)
    print("Metrics:")
    print(metrics_df)
    
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    #--------------------------------------- PLOTS -------------------------------------------
    #/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
    
    use_subset = True   # (T/F) to use subset or enterily of samples for the min, max and mean profiles 
    min_max_avg = True  # (T/F) to plot the min/max profiles using the mean value (Sum/24) or point by point
    sample_plot = 25     # Number of samples to plot for real and sintethic data
    
    plot_training_loss() # Plot the training loss from the CSV file

    plot_samples_by_month(synthetic_data, data, month=1 , use_subset=use_subset, min_max_avg=min_max_avg,
    num_samples=sample_plot) # min_max_avg [T=Num_samples+1(mean) / F=Num_samples+3(min, amx, mean)]
    
    plot_samples_all_months(synthetic_data, data, use_subset=use_subset, 
    min_max_avg=min_max_avg, num_samples=sample_plot)


if __name__ == "__main__":
    main()
