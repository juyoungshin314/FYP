from sklearn.decomposition import PCA
import numpy as np
import glob, os, random
import matplotlib.pyplot as plt

def load_data(root_path):

    data_list = []
    color_z = []
    data_list.extend(glob.glob(os.path.join(root_path, "*.npy")))
    random.shuffle(data_list)

    spin_m_data = np.load(data_list[0])

    for i in range(len(data_list)):
        color_z.append(float((data_list[i].split("\\")[-1]).split("_")[0])/10)

    for i in range(1, len(data_list)):
        tmp_data = np.load(data_list[i])
        spin_m_data = np.concatenate((spin_m_data, tmp_data), axis=0)

    return spin_m_data, color_z

def pca_data(data_path):
    spin_m_data, color_z = load_data(data_path)
    pca = PCA(n_components=2)
    pca.fit(spin_m_data)
    print(pca.explained_variance_ratio_)

    low_data = pca.transform(spin_m_data)

    return low_data, color_z

def pca_spin(data1, data2, data3, path1, path2, path3):
     data_list1 = []
     data_list2 = []
     data_list3 = []
     color_z = []
     data_list1.extend(glob.glob(os.path.join(path1, "*_50.npy")))
     data_list2.extend(glob.glob(os.path.join(path2, "*_50.npy")))
     data_list3.extend(glob.glob(os.path.join(path3, "*_50.npy")))
     pca = PCA(n_components=1)
     pca.fit(data1)
     v1 = pca.components_

     pca = PCA(n_components=1)
     pca.fit(data2)
     v2 = pca.components_

     pca = PCA(n_components=1)
     pca.fit(data3)
     v3 = pca.components_

     for i in range(len(data_list1)):
         color_z.append(float((data_list3[i].split("\\")[-1]).split("_")[0]))

     m1, m2, m3 = [], [], []
     for i in range(len(data_list1)):
         tmp_data1 = np.reshape(np.load(data_list1[i]), [-1, 1])
         m1.append(np.abs(np.dot(v1, tmp_data1)[0][0]/20))

         tmp_data2 = np.reshape(np.load(data_list2[i]), [-1, 1])
         m2.append(np.abs(np.dot(v2, tmp_data2)[0][0]/40))

         tmp_data3 = np.reshape(np.load(data_list3[i]), [-1, 1])
         m3.append(np.abs(np.dot(v3, tmp_data3)[0][0]/80))

