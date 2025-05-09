import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#----------------------------------- SAMPLES BY MONTH ------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

def plot_samples_by_month(synthetic_data, real_data, month, use_subset=False, min_max_avg=True, num_samples=50):
    # Create a figure with two subplots (left and right)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Synthetic Data Plot ---
    
    # Filter samples by the specified month
    filtered_synthetic = synthetic_data[synthetic_data['month'] == month]
    
    if torch.is_tensor(filtered_synthetic):
        filtered_synthetic = filtered_synthetic.numpy()
    
    # Extract hourly data from profiles
    synthetic_hourly_data = filtered_synthetic.iloc[:, :-1].values  # Exclude the 'month' column

    # Randomly select indices for the samples to plot
    num_samples_to_plot = min(num_samples, len(filtered_synthetic))
    selected_indices = np.random.choice(len(filtered_synthetic), size=num_samples_to_plot, replace=False)
    selected_data_plot = synthetic_hourly_data[selected_indices]
    # Data to calculate mean/min/max (depends on use_subset)
    selected_data_calc = selected_data_plot if use_subset else synthetic_hourly_data 

    # Caracteristic curves min, max, mean
    average_sample = np.mean(selected_data_calc, axis=0)
    
    if min_max_avg:
        # Min/Max based on the daily means using the selected samples
        daily_means = np.mean(selected_data_calc, axis=1)  
        max_idx = np.argmax(daily_means)
        min_idx = np.argmin(daily_means)

        max_profile = selected_data_calc[max_idx]
        min_profile = selected_data_calc[min_idx]
        
        # Delete duplicated curves only once if they are the same in the subset
        if use_subset:
            if max_idx == min_idx:
                selected_data_plot = np.delete(selected_data_plot, max_idx, axis=0)
            else:
                indices_to_delete = sorted([min_idx, max_idx], reverse=True)
                for idx in indices_to_delete:
                    selected_data_plot = np.delete(selected_data_plot, idx, axis=0)
    else:
        # Min/Max hour by hour generating new curves
        max_profile = np.max(selected_data_calc, axis=0)
        min_profile = np.min(selected_data_calc, axis=0)
    
    # Plot the synthetic data
    for curve in selected_data_plot:
        axes[0].plot(curve, color='gray', alpha=0.6)
    
    axes[0].plot(average_sample, label='Perfil Médio', color='green', linewidth=2)
    axes[0].plot(max_profile, label='Perfil Máximo', color='blue', linestyle='dashed', linewidth=2)
    axes[0].plot(min_profile, label='Perfil Mínimo', color='red', linestyle='dashed', linewidth=2)
    
    axes[0].set_title(f"Perfis Sintéticos para o Mês {month}")
    axes[0].set_xlabel("Hora")
    axes[0].set_ylabel("Geração PV")
    axes[0].set_ylim(0, pd.read_csv("PV_NY.csv")['Pnom'].iloc[0])
    axes[0].grid(True)
    axes[0].legend()

    # --- Real Data Plot ---
    
    # Filter samples by the specified month
    filtered_real = real_data[real_data['month'] == month]
    
    if torch.is_tensor(filtered_real):
        filtered_real = filtered_real.numpy()
    
    # Extract hourly data and calculate average, max, and min profiles
    real_hourly_data = filtered_real.iloc[:, :-1].values  # Exclude the 'month' column
    
    # Randomly select indices for the samples to plot
    num_samples_to_plot = min(num_samples, len(filtered_real))
    selected_indices = np.random.choice(len(filtered_real), size=num_samples_to_plot, replace=False)
    selected_data_plot = real_hourly_data[selected_indices]
    # Data to calculate mean/min/max (depends on use_subset)
    selected_data_calc = selected_data_plot if use_subset else real_hourly_data
    
    # Caracteristic curves min, max, mean
    average_sample = np.mean(selected_data_calc, axis=0)
    
    if min_max_avg:
        # Min/Max based on the daily means using the selected samples
        daily_means = np.mean(selected_data_calc, axis=1)  
        max_idx = np.argmax(daily_means)
        min_idx = np.argmin(daily_means)

        max_profile = selected_data_calc[max_idx]
        min_profile = selected_data_calc[min_idx]
        
        # Delete duplicated curves only once if they are the same in the subset
        if use_subset:
            if max_idx == min_idx:
                selected_data_plot = np.delete(selected_data_plot, max_idx, axis=0)
            else:
                indices_to_delete = sorted([min_idx, max_idx], reverse=True)
                for idx in indices_to_delete:
                    selected_data_plot = np.delete(selected_data_plot, idx, axis=0)
    else:
        # Min/Max hour by hour generating new curves
        max_profile = np.max(selected_data_calc, axis=0)
        min_profile = np.min(selected_data_calc, axis=0)
    
    # Plot the synthetic data
    for curve in selected_data_plot:
        axes[1].plot(curve, color='gray', alpha=0.6)
    
    axes[1].plot(average_sample, label='Perfil Médio', color='green', linewidth=2)
    axes[1].plot(max_profile, label='Perfil Máximo', color='blue', linestyle='dashed', linewidth=2)
    axes[1].plot(min_profile, label='Perfil Mínimo', color='red', linestyle='dashed', linewidth=2)
    
    axes[1].set_title(f"Perfis Reais para o Mês {month}")
    axes[1].set_xlabel("Hora")
    axes[1].set_ylabel("Geração PV")
    axes[1].set_ylim(0, pd.read_csv("PV_NY.csv")['Pnom'].iloc[0])
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#----------------------------------- ALL SAMPLES 12 -------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

def plot_samples_all_months(synthetic_data, real_data, num_samples=50, use_subset=False, min_max_avg=True):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    meses_portugues = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
    'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    
    for bloco in range(4):  # 4 grupos de 3 meses
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes[0, 0].set_title("Perfis Sintéticos", fontsize=14)
        axes[0, 1].set_title("Perfis Reais", fontsize=14)

        for i in range(3):
            month = bloco * 3 + i + 1
            nome_mes = f"{meses_portugues[month - 1]} - {month}"

            # ------------------------ Dados Sintéticos ------------------------
            ax_syn = axes[i, 0]
            filtered_synthetic = synthetic_data[synthetic_data['month'] == month]
            if torch.is_tensor(filtered_synthetic):
                filtered_synthetic = filtered_synthetic.numpy()
            synthetic_hourly = filtered_synthetic[:, :-1] if isinstance(filtered_synthetic, np.ndarray) else filtered_synthetic.iloc[:, :-1].values

            num_samples_to_plot = min(num_samples, len(synthetic_hourly))
            selected_indices = np.random.choice(len(synthetic_hourly), size=num_samples_to_plot, replace=False)
            selected_data_plot = synthetic_hourly[selected_indices]
            selected_data_calc = selected_data_plot if use_subset else synthetic_hourly

            avg_syn = np.mean(selected_data_calc, axis=0)

            if min_max_avg:
                daily_means = np.mean(selected_data_calc, axis=1)
                max_idx = np.argmax(daily_means)
                min_idx = np.argmin(daily_means)
                max_syn = selected_data_calc[max_idx]
                min_syn = selected_data_calc[min_idx]

                if use_subset:
                    if max_idx == min_idx:
                        selected_data_plot = np.delete(selected_data_plot, max_idx, axis=0)
                    else:
                        for idx in sorted([min_idx, max_idx], reverse=True):
                            selected_data_plot = np.delete(selected_data_plot, idx, axis=0)
            else:
                max_syn = np.max(selected_data_calc, axis=0)
                min_syn = np.min(selected_data_calc, axis=0)

            for curve in selected_data_plot:
                ax_syn.plot(curve, color='gray', alpha=0.5)
            ax_syn.plot(avg_syn, color='green', linewidth=2)
            ax_syn.plot(max_syn, color='blue', linestyle='dashed', linewidth=2)
            ax_syn.plot(min_syn, color='red', linestyle='dashed', linewidth=2)
            ax_syn.set_ylabel(nome_mes, fontsize=12)

            # ------------------------ Dados Reais ------------------------
            ax_real = axes[i, 1]
            filtered_real = real_data[real_data['month'] == month]
            if torch.is_tensor(filtered_real):
                filtered_real = filtered_real.numpy()
            real_hourly = filtered_real[:, :-1] if isinstance(filtered_real, np.ndarray) else filtered_real.iloc[:, :-1].values

            num_samples_to_plot = min(num_samples, len(real_hourly))
            selected_indices = np.random.choice(len(real_hourly), size=num_samples_to_plot, replace=False)
            selected_data_plot = real_hourly[selected_indices]
            selected_data_calc = selected_data_plot if use_subset else real_hourly

            avg_real = np.mean(selected_data_calc, axis=0)

            if min_max_avg:
                daily_means = np.mean(selected_data_calc, axis=1)
                max_idx = np.argmax(daily_means)
                min_idx = np.argmin(daily_means)
                max_real = selected_data_calc[max_idx]
                min_real = selected_data_calc[min_idx]

                if use_subset:
                    if max_idx == min_idx:
                        selected_data_plot = np.delete(selected_data_plot, max_idx, axis=0)
                    else:
                        for idx in sorted([min_idx, max_idx], reverse=True):
                            selected_data_plot = np.delete(selected_data_plot, idx, axis=0)
            else:
                max_real = np.max(selected_data_calc, axis=0)
                min_real = np.min(selected_data_calc, axis=0)

            for curve in selected_data_plot:
                ax_real.plot(curve, color='gray', alpha=0.5)
            ax_real.plot(avg_real, color='green', linewidth=2)
            ax_real.plot(max_real, color='blue', linestyle='dashed', linewidth=2)
            ax_real.plot(min_real, color='red', linestyle='dashed', linewidth=2)

            ax_real.set_ylabel("")

            ax_syn.set_ylim(0, pd.read_csv("PV_NY.csv")['Pnom'].iloc[0])
            ax_syn.set_xlim(0, 23)
            ax_real.set_ylim(0, pd.read_csv("PV_NY.csv")['Pnom'].iloc[0])
            ax_real.set_xlim(0, 23)
            ax_syn.grid(True)
            ax_real.grid(True)
            ax_syn.set_xticks([0, 6, 12, 18, 23])
            ax_real.set_xticks([0, 6, 12, 18, 23])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()