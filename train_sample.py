import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

def train_maf_with_loss_tracking(model, data, conditions, num_epochs=50, batch_size=64, learning_rate=1e-4, use_scheduler=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = None

    # Scheduler optional if lr don't decrease
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    losses = []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = data.shape[0] // batch_size

        indices = torch.randperm(data.shape[0])
        data = data[indices]
        conditions = conditions[indices]

        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            condition_batch = conditions[i * batch_size:(i + 1) * batch_size]

            loss = -model(batch, condition_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

        if use_scheduler:
            scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1

    # Save best model
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "maf_model.pt")
    print("******************************************************")
    print(f"Best model (epoch {best_epoch}, loss {best_loss:.6f}) saved to 'maf_model.pt'")
    print("******************************************************")
    print("Training complete!")
    print("******************************************************")

    return model, losses

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#----------------------------------- SAMPLING PROCESS ------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

def generate_synthetic_data(maf_model, scaler, num_samples=50):
    all_samples = []
    for month in range(1, 13):
        condition = torch.tensor([[float(month)]])
        new_samples = generate_samples(maf_model, scaler, condition, num_samples=num_samples)

        reshaped_samples = new_samples.reshape(-1, new_samples.shape[-1])
        month_column = torch.full((reshaped_samples.shape[0], 1), float(month))
        reshaped_samples_with_month = torch.cat((reshaped_samples, month_column), dim=1)

        all_samples.append(reshaped_samples_with_month.numpy())

    all_samples = torch.vstack([torch.tensor(month_samples) for month_samples in all_samples])
    
    return all_samples

#-------------------------------------------------------------------------------------------------

def generate_samples(model, scaler, condition, num_samples=5, threshold= pd.read_csv("PV_NY.csv")['Pnom'].iloc[0]):
    model.eval()
    valid_samples = []

    with torch.no_grad():
        while len(valid_samples) < num_samples:
            # Genera un bloque mayor para aumentar chance de válidos
            block_size = num_samples * 5
            samples = model.sample(block_size, condition=condition)

            # Escalado inverso y ReLU
            samples_np = samples.cpu().numpy().reshape(-1, 24)
            original_scale = scaler.inverse_transform(samples_np)
            original_scale = F.relu(torch.tensor(original_scale))

            # Filtrado vectorizado: solo acepta curvas con max <= threshold
            max_vals = torch.max(original_scale, dim=1).values
            mask = max_vals <= threshold
            filtered = original_scale[mask]

            # Añade muestras válidas al conjunto final
            if len(filtered) > 0:
                valid_samples.extend(filtered[:num_samples - len(valid_samples)])

    return torch.stack(valid_samples)

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#---------------------------------- PLOT TRAIN PROCESS ----------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

def plot_training_loss(csv_path="training_loss.csv"):
    """
    Load and plot training loss from a CSV file."""
    df = pd.read_csv(csv_path)
    plt.plot(df['epoch'], df['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()