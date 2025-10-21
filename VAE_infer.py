import matplotlib.pyplot as plt
import torch
from torch import nn
from model import VAE
from datasets import MCdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as cf
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from scipy.optimize import curve_fit
import numpy as np

modelpath = "{}_L_save_model".format(cf.L)
datapath = "{}_L".format(cf.L)

torch.set_grad_enabled(False)
model = VAE(cf.L, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(modelpath + "/vae_45.pth"))
dataset = MCdata(datapath)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#def inference():
#    tem, latent = [], []
#    for spin_m, t in tqdm(dataloader):
#        spin_m = torch.reshape(spin_m, (-1, 45*45))
#        _, out, _, _ = model(spin_m)
#        tem.append(t.numpy()[0])
#        latent.append(out.numpy()[0][0])

#    return tem, latent

def inference():
    temperature_values, latent_values, magnetization_values = [], [], []
    mag_by_temp = defaultdict(list)
    temp_by_temp = defaultdict(list)  # Store temperature values for each increment
    latent_by_temp = defaultdict(list)  # Store latent values for each increment

    for spin_m, t in tqdm(dataloader):
        spin_m = torch.reshape(spin_m, (1, cf.L*cf.L)).to(device)
        _, out, _, _ = model(spin_m)

        temp = t.numpy()[0]
        found_increment = False
        for i in np.arange(1, 3.7, 0.001):
            if i <= temp < i + 0.001:
                mag_by_temp[i].append(abs(out.numpy()[0][0]))
                temp_by_temp[i].append(temp)
                latent_by_temp[i].append(out.numpy()[0][0])

                temperature_values.append(t.numpy()[0])
                latent_values.append(out.numpy()[0][0])

                found_increment = True
                break
        if not found_increment:
            print(f"Warning: Temperature {temp} outside of increments.")

    avg_magnetization = {temp: np.mean(values) for temp, values in mag_by_temp.items()}
    avg_temperature = {temp: np.mean(values) for temp, values in temp_by_temp.items()}
    avg_latent = {temp: np.mean(values) for temp, values in latent_by_temp.items()}

    temperature_values_avg = list(avg_temperature.values())
    magnetization_values = list(avg_magnetization.values())
    latent_values_avg = list(avg_latent.values())

    return temperature_values, latent_values, magnetization_values, temperature_values_avg, latent_values_avg

if __name__ == "__main__":
    tem, latent, mag, tem_avg, latent_avg = inference()