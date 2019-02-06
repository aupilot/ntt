import csv
import torch
import torch.utils.data as data
from dataloader import NttTestDataset


model_file_list = [
    "./save/0205-1001cnn1 simple_fold_0/net-009-6.945.pkl",
    "./save/0205-1001cnn1 simple_fold_0/net-008-8.362.pkl",
]

class_list = ['MA_CH', 'MA_AD', 'MA_EL', 'FE_CH', 'FE_EL', 'FE_AD']

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    test_set = NttTestDataset()
    # test_set.num_samples=100        
    test_generator = data.DataLoader(test_set, **params)

    nets = []
    answer = list(dict())

    for model_file in model_file_list:
        cnn = torch.load(model_file, map_location=device)
        cnn.eval()  # Change model to 'eval' mode .
        nets.append(cnn)

    with torch.set_grad_enabled(False):
        for data, hash in test_generator:
            combined_classes = torch.zeros(6, device=device)
            for net in nets:
               # Here is the trick. The datagen generates batch of 1, but dataloader actually returns data in
                # batches with vaiable length. So we permutate dims to get a proper tensor
                outputs = net(data.permute((1,0,2)).to(device))
                classes = torch.softmax(outputs, 1).mean(0)
                combined_classes += classes
            winner = combined_classes.argmax().item()
            answer.append({'hash': hash[0], 'class': class_list[winner]})
            # print(winner)

    with open('answer.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=answer[0].keys(), delimiter='\t',lineterminator='\n')
        w.writeheader()
        w.writerows(answer)

