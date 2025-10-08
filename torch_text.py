import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np

TIME_STEPS = 288


def check_cuda_availability() -> None:
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA device count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('Current CUDA device:', torch.cuda.current_device())
        print('Current CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))


class Conv1dAutoEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3), 
            nn.ReLU(), 
            nn.Conv1d(32, 16, kernel_size=7, stride=2, padding=3), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=7, stride=2, padding=3, output_padding=1), 
            nn.ReLU(), 
            nn.ConvTranspose1d(16, 32, kernel_size=7, stride=2, padding=3, output_padding=1), 
            nn.ReLU(), 
            nn.ConvTranspose1d(32, input_channels, kernel_size=7, padding=3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


def test() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_cuda_availability()
    # This code from 'https://keras.io/examples/timeseries/timeseries_anomaly_detection/' is converted as torch version
    master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

    df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
    df_small_noise_url = master_url_root + df_small_noise_url_suffix
    df_small_noise = pd.read_csv(
        df_small_noise_url, parse_dates=True, index_col="timestamp"
    )

    df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
    df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
    df_daily_jumpsup = pd.read_csv(
        df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
    )

    print(df_small_noise.head())
    print(df_daily_jumpsup.head())

    fig, ax = plt.subplots()
    df_small_noise.plot(legend=False, ax=ax)
    plt.show()

    fig, ax = plt.subplots()
    df_daily_jumpsup.plot(legend=False, ax=ax)
    plt.show()

    # Normalize and save the mean and std we get,
    # for normalizing test data.
    training_mean = df_small_noise.mean()
    training_std = df_small_noise.std()
    df_training_value = (df_small_noise - training_mean) / training_std
    print("Number of training samples:", len(df_training_value))

    sequences = create_sequences(df_training_value.values)
    print("Training input shape: ", sequences.shape)

    x_train = torch.tensor(sequences, dtype=torch.float32).permute(0, 2, 1).to(device)
    print('x_train shape:', x_train.shape)

    model = Conv1dAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    batch_size = 128
    model.train()
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(x_train)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    # 학습 곡선 그리기
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.legend()
    plt.title("Training Loss")
    plt.show()

    # 재구성 손실 계산
    model.eval()
    with torch.no_grad():
        train_pred = model(x_train)
    train_mae_loss = torch.mean(torch.abs(train_pred - x_train), dim=[1,2]).cpu().numpy()

    plt.figure()
    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("Number of samples")
    plt.show()

    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold:", threshold)

    # 테스트 데이터 준비 및 이상 탐지
    df_test_norm = (df_daily_jumpsup - training_mean) / training_std
    test_seq = create_sequences(df_test_norm.values)
    print("Test sequences shape:", test_seq.shape)
    x_test = torch.tensor(test_seq, dtype=torch.float32).permute(0, 2, 1).to(device)

    with torch.no_grad():
        test_pred = model(x_test)
    test_mae_loss = torch.mean(torch.abs(test_pred - x_test), dim=[1,2]).cpu().numpy()

    plt.figure()
    plt.hist(test_mae_loss, bins=50)
    plt.xlabel("Test MAE loss")
    plt.ylabel("Number of samples")
    plt.show()

    anomalies = test_mae_loss > threshold
    print(f"Number of anomaly samples: {np.sum(anomalies)}")
    print(f"Index of anomaly samples: {np.where(anomalies)}")

    # 이상치 인덱스 이용해 시각화
    anomalous_indices = []
    for i in range(TIME_STEPS - 1, len(df_test_norm) - TIME_STEPS + 1):
        if np.all(anomalies[i - TIME_STEPS + 1:i]):
            anomalous_indices.append(i)

    df_subset = df_daily_jumpsup.iloc[anomalous_indices]
    fig, ax = plt.subplots()
    df_daily_jumpsup.plot(legend=False, ax=ax)
    df_subset.plot(legend=False, ax=ax, color="r")
    plt.show()


if __name__ == '__main__':
    test()
