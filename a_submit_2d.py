import csv
import time
import torch
import torch.utils.data as data
from dataloader import NttTestDataset, NttTestDataset3

model_file_list = [

    "./save/0215-13392D_fold_0/net-033-0.025.pkl",
    # "./save/0213-17472D_fold_1/net-015-0.068.pkl",
    # "./save/0213-16202D_fold_0/net-015-0.033.pkl",
]

class_list = ['MA_CH', 'MA_AD', 'MA_EL', 'FE_CH', 'FE_EL', 'FE_AD']

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    test_set = NttTestDataset3()

    # @@@@@@@@@@@@@@@@
    # test_set.num_samples=100
    test_generator = data.DataLoader(test_set, **params)

    nets = []
    answer = list(dict())

    for model_file in model_file_list:
        cnn = torch.load(model_file, map_location=device)
        cnn.eval()  # Change model to 'eval' mode .
        nets.append(cnn)

    with torch.set_grad_enabled(False):
        i = 0
        tick = time.time()
        for data, hash in test_generator:
            combined_classes = torch.zeros(6, device=device)
            for net in nets:
                # Here is the trick. The datagen generates batch of 1, but dataloader actually returns data in
                # batches with vaiable length. So we permutate dims to get a proper tensor
                # outputs = net(data.permute((1,0,2)).to(device))
                try:
                    a = data.shape[4]
                    data = data.squeeze(0)
                    data = data.to(device).type(torch.cuda.FloatTensor)
                except:
                    data = data.to(device).type(torch.cuda.FloatTensor)
                outputs = net(data)
                classes = torch.softmax(outputs, 1).mean(0)
                combined_classes += classes
            winner = combined_classes.argmax().item()
            answer.append({'hash': hash[0], 'class': class_list[winner]})
            # print(winner)
            i += 1
            if i % 100 == 0:
                tock = time.time()
                time_to_go = (len(test_generator)-i) / 100 * (tock - tick)
                print('Batch {:d} / {:d}, {:.1f} sec, to go: {:.0f} s'.format(
                    i,
                    len(test_generator),
                    tock - tick,
                    time_to_go)
                )
                tick = time.time()

    with open('answer.tsv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=answer[0].keys(), delimiter='\t',lineterminator='\n')
        w.writeheader()
        w.writerows(answer)

