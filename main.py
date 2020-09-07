import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# class definition
class LSTM(nn.Module):
    def __init__(self, Np, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.Np = Np

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, inMusic, hiddenStateFiltering):
        nSamples, batchSize, nFeatures = inMusic.shape[0], inMusic.shape[1], inMusic.shape[2]
        outMusic = torch.zeros((nSamples, batchSize, nFeatures, self.Np), dtype=torch.float).cuda()

        # outMusic[k, :, :, n] is \hat{x}_{k | j=k-1-n}
        # now, since j >= 0 we have that the term is defined only for k-1-n >= 0. Therefore for n > k-1 the term is not defined.
        # We therefore initialize it with the input values:

        for k in range(self.Np):
            for n in range(k, self.Np):
                outMusic[k:k+1, :, :, n:n+1] = inMusic[k:k+1, :, :, None]

        for k in range(inMusic.shape[0]):
            for n in range(min(k, self.Np)): # term is defined for n <= k-1
                if n == 0:
                    lstm_out, hiddenStateFiltering = self.lstm(inMusic[k:k+1], hiddenStateFiltering)
                    hiddenStatePrediction = (hiddenStateFiltering[0].clone(), hiddenStateFiltering[1].clone()) # h_{k+1 | k}
                    # input to the net is x_k and the state that was progressed in filtering mode
                    # the performed estimation is \hat{x}_{k+1 | k}
                    # and indeed outMusic[k + 0 + 1, :, :, 0] = \hat{x}_{k+1 | k}
                    if k == inMusic.shape[0] - 1: # last k
                        nextSampleEst = self.linear(lstm_out) # this corresponds with the output hiddenStateFiltering
                else:
                    lstm_out, hiddenStatePrediction = self.lstm(outMusic[k+n:k+n+1, :, :, n-1], hiddenStatePrediction)
                    # input to the net is \hat{x}_{k+n | k} and indeed outMusic[k + n, :, :, n - 1]  is \hat{x}_{k+n | k)}
                    # the performed estimation is \hat{x}_{k+n+1 | k} and indeed outMusic[k + n + 1, :, :, n] is \hat{x}_{k+n+1 | k }

                if k+n+1 < outMusic.shape[0]:
                    outMusic[k+n+1:k+n+2, :, :, n] = self.linear(lstm_out)

        return outMusic, hiddenStateFiltering, nextSampleEst

enableTrain = False
enableTest = True

PATH2SaveModel = './LSTM_Izi.pt'
enableSaveModel = True
enableFigures = True
enableConstantSin = True
batch_size = 20
num_epochs = 4 # 4000
nFutureSamples2Predict = 1

# Define model
print("Build LSTM RNN model ...")
num_layers, hidden_size = 2, 20
model = LSTM(Np=nFutureSamples2Predict, input_dim=1, hidden_dim=hidden_size, output_dim=1, num_layers=num_layers).cuda()
h_0 = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float).cuda()
c_0 = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float).cuda()

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
    minLoss = np.inf
    for epoch in range(num_epochs):
        chripDurations = minChirpDuration + torch.mul(torch.rand_like(chripDurations), maxChirpDuration - minChirpDuration)
        startFreqs = torch.mul(torch.rand_like(startFreqs), fs/2) # [hz/sec]
        startPhases = torch.mul(torch.rand_like(startPhases), two_pi)  # [rad/sec]

        k = torch.pow(4000/100, 1/chripDurations)  # k is the rate of exponential change in frequency to transform from 100 to 4000 hz in 10 seconds

        if enableConstantSin:
            if epoch == 0: freqFactor = tVec[:, None].repeat(1, k.shape[0])
        else:
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

        # zero out gradient, so they don't accumulate btw epochs
        model.zero_grad()

        modelPredictions, _, _ = model(noisySinWaves[:, :, None], (h_0, c_0))
        # modelPredictions[k, :, :, n] is \hat{x}_{k | j=k-1-n}

        loss = loss_function(modelPredictions, noisySinWaves[:, :, None, None].repeat(1, 1, 1, nFutureSamples2Predict)) # compute MSE loss
        loss.backward()  # (backward pass)
        optimizer.step()  # parameter update
        if loss.item() < minLoss:
            if enableSaveModel: torch.save(model.state_dict(), PATH2SaveModel)
            print(f'epoch {epoch}; model saved with loss {10*np.log10(loss.item()) - modelInputMean_dbW} [db]')
            minLoss = loss.item()
            if enableFigures and epoch == 0:
                plt.figure()
                plt.plot(tVec.cpu().numpy()[-1000:-1], noisySinWaves.detach().cpu().numpy()[-1000:-1,0])

        #print(f'epoch {epoch}; LSTM error to mean signal power ratio = {10*np.log10(loss.item()) - modelInputMean_dbW} [db]')

        if enableFigures and epoch == 0: # self loss calc
            selfCalcedMSE = np.mean(np.power(noisySinWaves[:, :, None, None].repeat(1, 1, 1, nFutureSamples2Predict).detach().cpu().numpy() - modelPredictions.detach().cpu().numpy(), 2))
            print(f'self calced MSE = {selfCalcedMSE}; pytorchCalcedMSE = {loss.item()}')

if enableTest:
    print("Testing ...")
    model.load_state_dict(torch.load(PATH2SaveModel))
    model.eval()
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

        modelPredictions, finalHiddenSingleSample, nextSampleEst = model(noisySinWaves[:, :, None], (h_0, c_0))

        loss = loss_function(modelPredictions, noisySinWaves[:, :, None, None].repeat(1, 1, 1, nFutureSamples2Predict))  # compute MSE loss
        print(f'Test: model has loss of {10 * np.log10(loss.item()) - modelInputMean_dbW} [db]')

        modelInput_W = np.mean(np.power(noisySinWaves.detach().cpu().numpy(), 2))
        modelOutput_W = np.mean(np.power(modelPredictions.detach().cpu().numpy(), 2))
        modelGains[epoch] = modelOutput_W/modelInput_W
        print(f'The gain of the model is {10*np.log10(modelOutput_W) - 10*np.log10(modelInput_W)} [db]')

        if enableFigures and epoch==0:
            plt.figure(figsize=(16, 4))
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.specgram(x=modelPredictions[:, i, 0, 0].detach().cpu().numpy(), NFFT=256, Fs=fs.cpu().numpy(), noverlap=128)
                plt.colorbar()
                plt.xlabel('sec')
                plt.ylabel('hz')
                plt.suptitle(r'Spectrograms of $\hat{x}_{k \mid k-1}$ @ Test')
            plt.show()

        modelPredictionsSequence = torch.empty_like(modelPredictions)
        # let modelPredictions[-1,:,:,0] be \hat{x}_{k | k-1} then nextSampleEst is \hat{x}_{k+1 | k}
        # finalHiddenSingleSample is h_{k+1 | k}
        modelPredictionsSequence[0, :, :, 0] = nextSampleEst    # \hat{x}_{k+1 | k}
        hidden = finalHiddenSingleSample                        # h_{k+1 | k}

        for i in range(1, tVec.shape[0]):
            modelPredictionsSequence[i, :, :, 0], hidden, _ = model(modelPredictionsSequence[i-1, :, :, 0][None, :, :], hidden)

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
                plt.suptitle('Spectrograms @ sequence test')
            plt.show()

    print(f'Mean model gain on all tests is {10*np.log10(np.mean(modelGains))} [db]')
