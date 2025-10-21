import matplotlib.pyplot as plt
import torch
from torch import nn
from model import VAE
from MCdata import MCdata  # Updated import
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as cf
from collections import defaultdict
from scipy.optimize import curve_fit
import numpy as np

modelpath = "{}_L_save_model".format(cf.L)
datapath = "{}_L".format(cf.L)

torch.set_grad_enabled(False)
model = VAE(cf.L, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(modelpath + "/vae_{}.pth".format(cf.L), weights_only=True))
dataset = MCdata(datapath)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def inference():
    temperature_values, latent_values, magnetization_values = [], [], []
    mag_by_temp = defaultdict(list)
    temp_by_temp = defaultdict(list)
    latent_by_temp = defaultdict(list)

    for spin_m, t in tqdm(dataloader):
        spin_m = torch.reshape(spin_m, (1, cf.L * cf.L * cf.L)).to(device)
        _, out, _, _ = model(spin_m)

        temp = t.numpy()[0]
        for i in np.arange(1.1, 9, 0.01):
            if i <= temp < i + 0.01:
                mag_by_temp[i].append(abs(out.numpy()[0][0]))
                temp_by_temp[i].append(temp)
                latent_by_temp[i].append(out.numpy()[0][0])
                temperature_values.append(t.numpy()[0])
                latent_values.append(out.numpy()[0][0])
                break
        else:
            print(f"Warning: Temperature {temp} outside of range 1.2 - 8.")

    avg_magnetization = {temp: np.mean(values) for temp, values in mag_by_temp.items()}
    avg_temperature = {temp: np.mean(values) for temp, values in temp_by_temp.items()}
    avg_latent = {temp: np.mean(values) for temp, values in latent_by_temp.items()}

    temperature_values_avg = list(avg_temperature.values())
    magnetization_values = list(avg_magnetization.values())
    latent_values_avg = list(avg_latent.values())

    print("Number of temperature bins:", len(avg_temperature))
    print("Temperature range:", min(temperature_values_avg), "to", max(temperature_values_avg))
    print("Sample temperatures:", temperature_values_avg[:5], "...")

    return temperature_values, latent_values, magnetization_values, temperature_values_avg, latent_values_avg

if __name__ == "__main__":
    tem, latent, mag, tem_avg, latent_avg = inference()
