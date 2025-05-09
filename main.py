import pandas as pd
import os, json, torch, joblib
import torch.nn.functional as F
from data_preparation import process_pv_data, prepare_data
from train_sample import train_maf_with_loss_tracking
from maf_model_2 import MAFModel_2


# Load the data and process it
pv_data_path = 'PV_NY.csv'  
process_pv_data(pv_data_path)

df_cleaned = pd.read_csv('data_profiles_labeled.csv', delimiter=',')

# Prepare the data
data_tensor, month_tensor, scaler = prepare_data(df_cleaned)

# Activación como aprámetros aceptables en JSON
activation_map = {"relu": F.relu, "leaky_relu": F.leaky_relu, "elu": F.elu,
    "selu": F.selu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "softplus": F.softplus}


#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#------------------------------------- MAF MODEL ----------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#Basic Parameters
num_inputs = data_tensor.shape[1] # Number of features (24 hours + 1 month)
condition_size = 1                # Size of the condition (month)
num_hidden = 128                   # Number of hidden units (neurons) in the MAF model
num_camadas = 2                    # Number of blocks in each autoregressive layer (share the same hidden and layers)
num_layers = 4                     # Number of autoregressive layers in the MAF model

residual = True                      # Residual connections in the MAF model? (T/F)
rndm_mask = False                    # Random masking in the MAF model? (T/F) order of the dependencies aleatorie?
norm_batch = False                   # Use batch normalization in the MAF model? (T/F) Normalization of the activation?
drop_prob = 0.0                      # Dropout probability in the MAF model (0.0 to 1.0)
fun_act = "relu"                     # Activation function in the MAF model (relu,leaky_relu_elu,selu,tanh,sigmoid,softplus)

maf_model = MAFModel_2(num_inputs=num_inputs, num_hidden=num_hidden, num_layers=num_layers, condition_size=condition_size,
    num_camadas=num_camadas, residual=residual, rndm_mask=rndm_mask, fun_act=activation_map[fun_act],
    drop_prob=drop_prob, norm_batch=norm_batch)
params = {"num_inputs": num_inputs, "num_hidden": num_hidden, "num_layers": num_layers, "condition_size": condition_size,
    "num_camadas": num_camadas, "residual": residual, "rndm_mask": rndm_mask, "fun_act": fun_act,
    "drop_prob": drop_prob, "norm_batch": norm_batch}

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#----------------------------------- TRAIN & SAMPLE --------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

epochs = 300          # Number of epochs to train the MAF model
batch = 64            # Batch size for training the MAF model
learn = 1e-4          # Learning rate for the optimizer
use_scheduler= False  # Use learning rate scheduler? (T/F) to vary the learning rate during training

# Train the model
maf_model, losses = train_maf_with_loss_tracking(maf_model, data_tensor, month_tensor,
    num_epochs=epochs, batch_size=batch, learning_rate=learn, use_scheduler=use_scheduler)

# Save Prob-log losses by epoch in CSV
loss_df = pd.DataFrame({'epoch': list(range(1, len(losses) + 1)), 'loss': losses})
loss_df.to_csv('training_loss.csv', index=False)
print("Training losses saved to --training_loss.csv--")
print("******************************************************")

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#----------------------------------- SAVE THE MODEL --------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

#Remove samples generated in other cases
csv_file_path = 'synthetic_data_profiles.csv'
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)
    print(f'Old synthetic data profiles deleted --{csv_file_path}--')
    print("******************************************************")

#Save the sacler
joblib.dump(scaler, "scaler.pkl")

# Guardar pesos del modelo
torch.save(maf_model.state_dict(), "maf_model.pt")

# Guardar hiperparámetros en JSON
with open("maf_model_config.json", "w") as f:
    json.dump(params, f)




