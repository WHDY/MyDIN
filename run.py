import os
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataGenerator import dataGenerator
from model import DIN
from config import Config

if __name__ == "__main__":
    # parameters
    config = Config()

    # cuda environments
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    dev = torch.device(config.cuda) if torch.cuda.is_available() else torch.device('cpu')

    # ============== load data ============== #
    dataset = dataGenerator(dataPath=config.dataPath, ratingBinThreshold=config.ratingBinThreshold,
                            maxSequenceLen=config.maxSequenceLen, splitRatio=config.splitRatio)
    # train dataset
    trainRowData, trainLabel = map(torch.tensor, (dataset.trainRowData, dataset.trainLabel))
    trainDataLoader = DataLoader(dataset=TensorDataset(trainRowData, trainLabel),
                                 batch_size=config.batchSize, shuffle=True)
    # test dataset
    testRowData = [torch.tensor(x) for x in dataset.testRowData]
    testLable = [torch.tensor(x) for x in dataset.testLabel]
    # user features and movie features
    userFeatures = torch.tensor(dataset.userFeatures).to(dev)
    movieFeatures = torch.tensor(dataset.movieFeatures).to(dev)

    # ============= load model =============== #
    # embedding layer info:
    # embeddingGroups[embeddingGroupName] = (vocabulary size, embedding size)
    model = DIN(embeddingGroupInfo=config.embeddingGroups,
                MLPInfo=config.MLPInfo,
                attMLPInfo=config.AttMLPInfo,
                isUseBN=config.isUseBN,
                l2RegEmbedding=config.l2RegEmbedding,
                dropoutRate=config.dropoutRate,
                initStd=config.initStd,
                device=dev)

    # ===== optimization & loss function ===== #
    optimizer = config.optimizer(model.parameters(), lr=config.learningRate)

    lrSchedule = config.lrSchedule(optimizer, config.decay)

    lossFunc = config.lossFunc

    # ============== metrics ================= #
    metricFunc = config.metricFunc

    # =============== train ================== #
    for E in range(config.epoch):
        print("epoch{}: ".format(E + 1))
        model.train()
        epochLoss = 0
        for data, label in tqdm(trainDataLoader):
            data, label = data.to(dev), label.to(dev)

            loss = model.loss(data, userFeatures, movieFeatures, label, lossFunc)

            epochLoss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("epoch {} loss: {}".format(E + 1, epochLoss))

        with torch.no_grad():
            model.eval()
            impressionNum = 0.0
            impressAuc = 0.0
            for k in range(len(testRowData)):
                data, label = testRowData[k].to(dev), testLable[k]
                preds = model.predict(data, userFeatures, movieFeatures)
                auc = metricFunc(label, preds)

                impressAuc += testLable[k].shape[0] * auc
                impressionNum += testLable[k].shape[0]

            print("epoch {} evaluation auc: {}".format(E + 1, impressAuc / impressionNum))

        print('Epoch-{0} lr: {1}'.format(E + 1, optimizer.param_groups[0]['lr']))
        if (E + 1) % config.decayStep == 0:
            lrSchedule.step()
