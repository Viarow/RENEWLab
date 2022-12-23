import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import sys
import yaml
from tqdm import tqdm
from langevine.utils.util import dict2namespace
from langevine.model.remimo.sample_generator import sample_generator
from langevine.model.remimo.iterative_classifier import iterative_classifier
from langevine.model.remimo.utils_remimo import *

def get_snr_range(NT):
    peak = NT*(5.0/16.0) + 6.0
    snr_low = peak
    snr_high = peak+10.0
    return (snr_low, snr_high)


def validate_model_given_data(model, validtn_H, validtn_y, validtn_j_indices, validtn_noise_sigma, device, criterion=None):
    with torch.no_grad():

        validtn_H = validtn_H.to(device=device).float()
        validtn_y = validtn_y.to(device=device).float()
        validtn_noise_sigma = validtn_noise_sigma.to(device=device).float()
        validtn_out = model.forward(validtn_H, validtn_y, validtn_noise_sigma)

        if (criterion):
            validtn_j_indices = validtn_j_indices.to(device=device)
            loss = loss_function(criterion, validtn_out, validtn_j_indices, nlayers)
            validtn_j_indices = validtn_j_indices.to(device='cpu')

        validtn_out = validtn_out[-1].to(device='cpu')
        accr = accuracy(validtn_out, validtn_j_indices)

        del validtn_H, validtn_y, validtn_noise_sigma, validtn_out, validtn_j_indices

        if (criterion):
            return accr, loss.item()
        else:
            return accr, None

def mini_validation(model, mini_validation_dict, i, device, criterion=None, save_to_file=True):
    result_dict = {int(NT):{} for NT in validtn_NT_list}
    loss_list = []
    for index,NT in enumerate(validtn_NT_list):
        for snr in snrdb_list[NT]:
            big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = mini_validation_dict[NT][snr]
            accr, loss = validate_model_given_data(model, big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma, device, criterion)
            result_dict[NT][snr] = accr
            loss_list.append(loss*factor_list[index])

    print('Validtn result, Accr for 16 : ', result_dict[16])
    print('Validation resut, Accr for 32 : ', result_dict[32])
    if (save_to_file):
        with open(curr_accr, 'w') as f:
            print((i, result_dict), file=f)
        print('Saved intermediate validation results at : ', curr_accr)

    if (criterion):
        return np.sum(loss_list)

def generate_big_validtn_data(generator, batch_size, corr_flag, rho, batch_corr, rho_low, rho_high):
    validtn_data_dict = {int(NT):{} for NT in validtn_NT_list}
    for NT in validtn_NT_list:
        for snr in tqdm(snrdb_list[NT]):
            big_validtn_H, big_validtn_y, big_validtn_j_indices, big_noise_sigma = generator.give_batch_data(int(NT), snr_db_min=snr, snr_db_max=snr, batch_size=batch_size, correlated_flag=corr_flag, rho=rho, batch_corr=batch_corr,rho_low=rho_low, rho_high=rho_high)
            validtn_data_dict[int(NT)][snr] = (big_validtn_H, big_validtn_y , big_validtn_j_indices, big_noise_sigma)
    print("validtn data dict generated")
    return validtn_data_dict

def save_model_func(model, optimizer, model_filename):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_filename)
    print('******Model Saved********** at directory : ', model_filename)


def train(model, optimizer, lr_scheduler, data_dict, device, config):
    
    criterion = nn.CrossEntropyLoss().to(device=device)
    model.train()

    H_set = data_dict['H']
    y_set = data_dict['y']
    j_indices_set = data_dict['j_indices']
    noise_sigma_set = data_dict['noise_sigma']
    num_batches = H_set.shape[0]

    for epoch in range(config.num_epochs):
        running_loss = 0.0

        for i in range(num_batches):
            H = H_set[i].to(device=device)
            y = y_set[i].to(device=device)
            noise_sigma = noise_sigma_set[i].to(device=device)
            j_indices = j_indices_set[i].to(device=device)

            out = model.forward(H, y, noise_sigma)
            del H, y, noise_sigma

            loss = loss_function(criterion, out, j_indices, config.nlayers)
            del j_indices, out

            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss

            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    print("training finished.")
    if config.model_path:
        save_model_func(model, optimizer, config.model_path)


def main():
    dirPath = os.getcwd()
    with open(dirPath + '/langevine/remimo_config.yml', 'r') as f:
        aux = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(aux)
    SEED = 123
    torch.manual_seed(SEED)
    useGPU = True
    if useGPU and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    data_dict = torch.load(config.data_path)
    generator = sample_generator(config.train_batch_size, config.mod_n, config.NR)
    model = iterative_classifier(config.d_model, config.n_head, config.nhid, config.nlayers, config.mod_n, config.NR, config.d_transmitter_encoding, generator.real_QAM_const, generator.imag_QAM_const, generator.constellation, device, config.dropout)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', 0.91, 0, 0.0001, 'rel', 0, 0, 1e-08, verbose = True)

    print("start training")
    train(model, optimizer, lr_scheduler, data_dict, device, config)


if __name__ == '__main__':
    main()