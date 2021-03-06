import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from musicFunctions import *

# class definition
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, inMusic, hidden):
        lstm_out, hidden = self.lstm(inMusic, hidden)
        outMusic = self.linear(lstm_out)
        return outMusic, hidden, lstm_out



enableTrain = False
enableTest = True

PATH2SaveModel = './GRU_Izi.pt'
enableSaveModel = False
enableFigures = True
enableConstantSin = True
batch_size = 20
num_epochs = 4000
numberOfFutureSamples = 25

# Define model
print("Build RNN model ...")
num_layers, hidden_size = 1, 20
model = GRU(input_dim=1, hidden_dim=hidden_size, output_dim=1, num_layers=num_layers).cuda()


loss_function = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

fs = torch.tensor(8000, dtype=torch.float).cuda() # [hz]
sequenceDuration = 2 # sec
SNR = 30 # [db]
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

modelInputMean_dbW = 10*np.log10(np.mean(np.power(noisySinWaves.cpu().numpy(), 2)))

'''
plt.plot(tVec.cpu().numpy(), noisySinWaves[:, 0].cpu().numpy())
plt.xlabel('sec')
plt.title('LSTM input wave')
plt.grid(True)
plt.show()
'''
if enableTrain:
    print("Training ...")
    model.train()
    trainingFunc(model, loss_function, num_layers, batch_size, hidden_size, optimizer, enableSaveModel, PATH2SaveModel,
                 modelInputMean_dbW, numberOfFutureSamples, chripDurations, startFreqs, startPhases, num_epochs,
                 minChirpDuration, maxChirpDuration, fs, two_pi, enableConstantSin, tVec, noiseStd, enableFigures, enableLearn=True)


if enableTest:
    print("Testing ...")

    h_0 = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float).cuda()
    model.load_state_dict(torch.load(PATH2SaveModel))
    model.eval()

    # calling the train once to calculate the loss on all future predictions (in test the loss is calculated only for a single future prediction)
    trainingFunc(model, loss_function, num_layers, batch_size, hidden_size, optimizer, False, PATH2SaveModel,
                 modelInputMean_dbW, numberOfFutureSamples, chripDurations, startFreqs, startPhases, 1,
                 minChirpDuration, maxChirpDuration, fs, two_pi, enableConstantSin, tVec, noiseStd, enableFigures, enableLearn=False)

    modelGains = np.zeros(num_epochs)
    num_epochs = 1
    for epoch in range(num_epochs):

        chripDurations = minChirpDuration + torch.mul(torch.rand_like(chripDurations), maxChirpDuration - minChirpDuration)
        startFreqs = torch.mul(torch.rand_like(startFreqs), fs / 2)  # [hz/sec]
        startPhases = torch.mul(torch.rand_like(startPhases), two_pi)  # [rad/sec]

        k = torch.pow(4000 / 100, 1 / chripDurations)  # k is the rate of exponential change in frequency to transform from 100 to 4000 hz in 10 seconds

        if enableConstantSin:
            if epoch == 0: freqFactor = tVec[:, None].repeat(1, k.shape[0])
        else:
            freqFactor = torch.true_divide(torch.pow(k[None, :].repeat(tVec.shape[0], 1), tVec[:, None].repeat(1, k.shape[0])) - 1, torch.log(k)[None, :].repeat(tVec.shape[0], 1))

        phases = torch.mul(torch.mul(startFreqs[None, :].repeat(tVec.shape[0], 1), two_pi), freqFactor) + startPhases[None, :].repeat(tVec.shape[0], 1)
        pureSinWaves = torch.sin(phases)
        noise = torch.mul(torch.randn_like(pureSinWaves), noiseStd)
        noisySinWaves = pureSinWaves + noise

        modelPredictions, finalHiddenSingleSample, _ = model(noisySinWaves[:, :, None], h_0)

        loss = loss_function(modelPredictions[:-1], noisySinWaves[1:, :, None])  # compute MSE loss
        print(f'Test: model has loss of {10 * np.log10(loss.item()) - modelInputMean_dbW} [db]')

        modelInput_W = np.mean(np.power(noisySinWaves.detach().cpu().numpy(), 2))
        modelOutput_W = np.mean(np.power(modelPredictions.detach().cpu().numpy(), 2))
        modelGains[epoch] = modelOutput_W/modelInput_W
        print(f'The gain of the model is {10*np.log10(modelOutput_W) - 10*np.log10(modelInput_W)} [db]')

        if enableFigures and epoch==0:
            plt.figure(figsize=(16, 4))
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.specgram(x=modelPredictions[:, i, 0].detach().cpu().numpy(), NFFT=256, Fs=fs.cpu().numpy(), noverlap=128)
                plt.colorbar()
                plt.xlabel('sec')
                plt.ylabel('hz')
                plt.suptitle('Spectrograms @ Test')
            plt.show()

        modelPredictionsSequence = torch.empty_like(modelPredictions)
        modelPredictionsSequence[0], hidden, _ = model(modelPredictions[-1][None, :, :], finalHiddenSingleSample)
        for i in range(1, tVec.shape[0]):
            modelPredictionsSequence[i], hidden, _ = model(modelPredictionsSequence[i-1][None, :, :], hidden)

        if enableFigures and epoch==0:
            plt.figure(figsize=(16, 8))
            for i in range(4):
                signal = torch.cat((modelPredictions, modelPredictionsSequence), dim=0)[:, i, 0].detach().cpu().numpy()
                tVecSignal = (1/fs.cpu().numpy()) * np.arange(0, signal.shape[0])

                plt.subplot(4, 2, 2*i + 1)
                plt.specgram(x=signal, NFFT=256, Fs=fs.cpu().numpy(), noverlap=128)
                plt.colorbar()
                plt.xlabel('sec')
                plt.ylabel('hz')

                plt.subplot(4, 2, 2*i + 2)
                minIdx, maxIdx = int(np.round(tVecSignal.shape[0]/2) - 100), int(np.round(tVecSignal.shape[0]/2) + 100)
                plt.plot(tVecSignal[minIdx:maxIdx], signal[minIdx:maxIdx])
                plt.xlabel('sec')
                plt.grid(True)
                plt.suptitle(r'Spectrograms of $\hat{x}_{k \mid \operatorname{min}(k-1, \frac{N}{2})}$')
            plt.show()

    print(f'Mean model gain on all tests is {10*np.log10(np.mean(modelGains))} [db]')
