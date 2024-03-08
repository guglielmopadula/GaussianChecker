import torch
import numpy as np
import meshio
from tqdm import trange
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.utils.extmath import randomized_svd
import numpy as np  
import scipy.stats
import matplotlib.pyplot as plt




x=np.random.laplace(size=(600,5))
data=x+1/20*x**(3)
data=(np.random.rand(3000,5)@data.T).T

pca=PCA(n_components=100)
pca.fit(data)
print(np.linalg.matrix_rank(data))
print(np.linalg.norm(data-pca.inverse_transform(pca.transform(data)))/np.linalg.norm(data))
data=pca.transform(data)

out_size=data.shape[1]
data=torch.tensor(data,dtype=torch.float32)

from revnet import RevNet
model=RevNet(5,out_size)



optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
num_epochs=5000
for epoch in trange(num_epochs):
    optimizer.zero_grad()
    recon=model(model.inverse(data)) 
    loss = torch.linalg.norm(recon-data)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        loss_tmp=torch.mean(torch.linalg.norm(model(model.inverse(data))-data,axis=1)/torch.linalg.norm(data,axis=1))

model.eval()
with torch.no_grad():
    latent_data=model.inverse(data)
    np.save('latent_data.npy',latent_data)



latent_data=np.load('latent_data.npy')
def check_gaussian(dataset, alpha, n_components=None):
    n=dataset.shape[0]
    #dataset: numpy array of size n_samples x n_features
    #alpha: float, the level of significance
    #n_components: int, the number of principal components to keep
    #Returns: True if the dataset is Gaussian, False otherwise
    if n_components is None:
        n_components = np.linalg.matrix_rank(dataset)
    mu=np.mean(dataset,axis=0)
    centered_data=dataset-mu
    barX=centered_data/np.sqrt(dataset.shape[0]-1)
    u,s,vh=randomized_svd(barX.T,n_components=n_components)
    barBplus=np.diag((1/s))@(u.T)
    Z=centered_data@barBplus.T
    eps=np.sqrt(np.log(2/alpha)/(2*n))
    flag=True
    for i in range(n_components):
        Z_i=Z[:,i]
        y=np.linspace(Z_i.min()-eps,Z_i.max()+eps,10000)
        true=scipy.stats.laplace.cdf(y)
        pred=np.sum(Z_i.reshape(-1,1)<=y.reshape(1,-1),axis=0)/600
        plt.plot(y,pred)
        plt.plot(y,true+eps)
        plt.plot(y,true-eps)
        plt.plot(y,true)
        plt.show()
        if not np.prod(np.abs(pred-true)<eps):
            flag=False  
            break

    return flag

print(check_gaussian(latent_data,0.05))