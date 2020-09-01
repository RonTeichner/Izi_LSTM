import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, inMusic):
        lstm_out, hidden = self.lstm(inMusic)
        outMusic = self.linear(lstm_out)
        return outMusic

enableFigures = True
batch_size = 35
num_epochs = 400

# Define model
print("Build LSTM RNN model ...")
model = LSTM(input_dim=1, hidden_dim=5, batch_size=batch_size, output_dim=1, num_layers=1).cuda()

loss_function = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

fs = torch.tensor(8000, dtype=torch.float).cuda() # [hz]
sequenceDuration = 10 # sec
SNR = 15 # [db]
# mean power of sin signal per sample is 0.5
signalPowerPerSample = 10*np.log10(0.5)
noisePowerPerSample_dbW = signalPowerPerSample - SNR
noisePowerPerSample_W = np.power(10, (noisePowerPerSample_dbW/10))
noiseStd = torch.tensor(np.sqrt(noisePowerPerSample_W), dtype=torch.float).cuda()
minChirpDuration = torch.tensor(6, dtype=torch.float).cuda() # sec
maxChirpDuration = torch.tensor(12, dtype=torch.float).cuda() # sec
# declaring variables and sending to cuda:
two_pi = torch.tensor(2*np.pi, dtype=torch.float).cuda()
tVec = torch.arange(0, sequenceDuration, 1/fs, dtype=torch.float).cuda()

chripDurations = (minChirpDuration + torch.mul(torch.rand(batch_size, dtype=torch.float), maxChirpDuration - minChirpDuration)).cuda() # sec
startFreqs = torch.mul(torch.rand(batch_size, dtype=torch.float), fs/2).cuda() # [hz/sec]
startPhases = torch.mul(torch.rand(batch_size, dtype=torch.float), two_pi).cuda()  # [rad/sec]

k = torch.pow(4000/100, 1/chripDurations)  # k is the rate of exponential change in frequency to transform from 100 to 4000 hz in 10 seconds
freqFactor = torch.true_divide(torch.pow(k[None, :].repeat(tVec.shape[0], 1), tVec[:, None].repeat(1, k.shape[0])) - 1, torch.log(k)[None, :].repeat(tVec.shape[0], 1))

phases = torch.mul(torch.mul(startFreqs[None, :].repeat(tVec.shape[0], 1), two_pi), freqFactor) + startPhases[None, :].repeat(tVec.shape[0], 1)
pureSinWaves = torch.sin(phases)
noise = torch.mul(torch.randn_like(pureSinWaves), noiseStd)
noisySinWaves = pureSinWaves + noise


'''
plt.plot(tVec.cpu().numpy(), noisySinWaves[:, 0].cpu().numpy())
plt.xlabel('sec')
plt.title('LSTM input wave')
plt.grid(True)
plt.show()
'''
print("Training ...")
for epoch in range(num_epochs):
    chripDurations = minChirpDuration + torch.mul(torch.rand_like(chripDurations), maxChirpDuration - minChirpDuration)
    startFreqs = torch.mul(torch.rand_like(startFreqs), fs/2) # [hz/sec]
    startPhases = torch.mul(torch.rand_like(startPhases), two_pi)  # [rad/sec]

    k = torch.pow(4000/100, 1/chripDurations)  # k is the rate of exponential change in frequency to transform from 100 to 4000 hz in 10 seconds
    freqFactor = torch.true_divide(torch.pow(k[None, :].repeat(tVec.shape[0], 1), tVec[:, None].repeat(1, k.shape[0])) - 1, torch.log(k)[None, :].repeat(tVec.shape[0], 1))

    phases = torch.mul(torch.mul(startFreqs[None, :].repeat(tVec.shape[0], 1), two_pi), freqFactor) + startPhases[None, :].repeat(tVec.shape[0], 1)
    pureSinWaves = torch.sin(phases)
    noise = torch.mul(torch.randn_like(pureSinWaves), noiseStd)
    noisySinWaves = pureSinWaves + noise

    if enableFigures and epoch == 0:
        noiseEmpiricPower_dbW = 10 * np.log10(np.mean(np.power(noise.cpu().numpy(), 2)))
        signalEmpiricPower_dbW = 10 * np.log10(np.mean(np.power(pureSinWaves.cpu().numpy(), 2)))
        empiricalSNR = signalEmpiricPower_dbW - noiseEmpiricPower_dbW
        print(f'Empirical SNR: {empiricalSNR}')

        plt.figure(figsize=(16, 4))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.specgram(x=noisySinWaves[:, i].cpu().numpy(), NFFT=256, Fs=fs.cpu().numpy(), noverlap=128)
            plt.colorbar()
            plt.xlabel('sec')
            plt.ylabel('hz')
            plt.suptitle('Spectrograms')
        plt.show()

    # Init hidden state - if you don't want a stateful LSTM (between epochs)
    model.hidden = model.init_hidden()

    # zero out gradient, so they don't accumulate btw epochs
    model.zero_grad()

    modelPredictions = model(noisySinWaves[:, :, None])
    loss = loss_function(modelPredictions[:-1], noisySinWaves[1:, :, None]) # compute MSE loss
    loss.backward()  # (backward pass)
    optimizer.step()  # parameter update

    print(f'epoch {epoch}; LSTM error to noise ratio = {10*np.log10(loss.item()) - noisePowerPerSample_dbW} [db]')

    if enableFigures and epoch == 0: # self loss calc
        selfCalcedMSE = np.mean(np.power(noisySinWaves[1:, :, None].detach().cpu().numpy() - modelPredictions[:-1].detach().cpu().numpy(), 2))
        print(f'self calced MSE = {selfCalcedMSE}; pytorchCalcedMSE = {loss.item()}')
    x=3
