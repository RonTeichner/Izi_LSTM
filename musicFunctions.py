import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time


def trainingFunc(model, loss_function, num_layers, batch_size, hidden_size, optimizer, enableSaveModel, PATH2SaveModel, modelInputMean_dbW, numberOfFutureSamples, chripDurations, startFreqs, startPhases, num_epochs, minChirpDuration, maxChirpDuration, fs, two_pi, enableConstantSin, tVec, noiseStd, enableFigures, enableLearn=False):
    minLoss = np.inf
    for epoch in range(num_epochs):
        chripDurations = minChirpDuration + torch.mul(torch.rand_like(chripDurations), maxChirpDuration - minChirpDuration)
        startFreqs = torch.mul(torch.rand_like(startFreqs), fs / 2)  # [hz/sec]
        startPhases = torch.mul(torch.rand_like(startPhases), two_pi)  # [rad/sec]

        k = torch.pow(4000 / 100,
                      1 / chripDurations)  # k is the rate of exponential change in frequency to transform from 100 to 4000 hz in 10 seconds

        if enableConstantSin:
            if epoch == 0: freqFactor = tVec[:, None].repeat(1, k.shape[0])
        else:
            freqFactor = torch.true_divide(
                torch.pow(k[None, :].repeat(tVec.shape[0], 1), tVec[:, None].repeat(1, k.shape[0])) - 1,
                torch.log(k)[None, :].repeat(tVec.shape[0], 1))

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
                plt.suptitle(r'Spectrograms of $x_k$')
            plt.show()

        # zero out gradient, so they don't accumulate btw epochs
        if enableLearn: model.zero_grad()

        if numberOfFutureSamples == 1:
            h_0 = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float).cuda()
            modelPredictions, _, _ = model(noisySinWaves[:, :, None], h_0)
            loss = loss_function(modelPredictions[:-1], noisySinWaves[1:, :, None])  # compute MSE loss
        else:
            nSamples, nFeatures = noisySinWaves.shape[0], 1
            loss = torch.zeros(1, dtype=torch.float).cuda()
            if not enableLearn: lossOnLastPrediction = torch.zeros(1, dtype=torch.float).cuda()
            if num_layers > 1:
                modelPredictions = torch.zeros(1, nSamples, nFeatures, dtype=torch.float).cuda()
                hiddenStates = torch.zeros(num_layers, nSamples, hidden_size, dtype=torch.float).cuda()

            filteringPredictions, _, allFilteringStates = model(noisySinWaves[:, :, None], torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float).cuda())
            filteringPredictions, allFilteringStates = filteringPredictions.transpose(1, 0).contiguous(), allFilteringStates.transpose(1, 0).contiguous()
            for b in range(batch_size):
                signalStartTime = time.time()
                singleSignal = noisySinWaves[:, b:b + 1, None]
                if num_layers > 1: hidden = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float).cuda()
                for n in range(numberOfFutureSamples):
                    if n == 0:
                        if num_layers > 1:
                            for k in range(nSamples):
                                modelPredictions[:, k:k + 1, :], hidden, _ = model(singleSignal[k:k + 1], hidden)
                                hiddenStates[:, k:k + 1, :] = hidden
                        else:
                            # filteringPredictions, _, allFilteringStates = model(singleSignal, hidden)
                            # modelPredictions = filteringPredictions.transpose(1, 0)
                            # hiddenStates = allFilteringStates.transpose(1, 0)
                            modelPredictions = filteringPredictions[b:b + 1]
                            hiddenStates = allFilteringStates[b:b + 1]
                    else:
                        modelPredictions, hiddenStates, _ = model(modelPredictions, hiddenStates)
                    # modelPredictions is \hat{x}_{n+1:n+N} while singleSignal is x_{0:N-1}
                    # the index of time N-1 in modelPredictions is -1-n-1 and the index of time n+1 in singleSignal is n+1
                    loss = torch.add(loss, torch.true_divide(loss_function(modelPredictions[:, :-1 - n], singleSignal[n + 1:].transpose(1, 0)), batch_size * numberOfFutureSamples))  # compute MSE loss
                    if not enableLearn:
                        if n == numberOfFutureSamples-1:
                            lossOnLastPrediction = torch.add(lossOnLastPrediction, torch.true_divide(loss_function(modelPredictions[:, :-1 - n], singleSignal[n + 1:].transpose(1, 0)), batch_size))  # compute MSE loss
                    '''
                    if n == 1:
                        signalEndFilteringTime = time.time()
                        print(f'Epoch {epoch}: Training: signal no. {b} filtering time: {(signalEndFilteringTime - signalStartTime)/1e-3} [ms]')
                    elif n > 1:
                        signalEndTime = time.time()
                        print(f'Epoch {epoch}: Training: signal no. {b} prediction time: {(signalEndTime-signalEndFilteringTime)/1e-3} [ms]')
                    '''
        if enableLearn:
            loss.backward()  # (backward pass)
            optimizer.step()  # parameter update
        if loss.item() < minLoss:
            if enableSaveModel:
                torch.save(model.state_dict(), PATH2SaveModel)
                print(f'epoch {epoch}; model saved with loss {10 * np.log10(loss.item()) - modelInputMean_dbW} [db]')
            else:
                print(f'epoch {epoch}; model has loss {10 * np.log10(loss.item()) - modelInputMean_dbW} [db]')
            if not enableLearn:
                print(f'epoch {epoch}; model has loss {10 * np.log10(lossOnLastPrediction.item()) - modelInputMean_dbW} [db] on final prediction')
            minLoss = loss.item()
            if enableFigures and epoch == 0:
                plt.figure()
                plt.plot(tVec.cpu().numpy()[-1000:-1], noisySinWaves.detach().cpu().numpy()[-1000:-1, 0])