# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:05:49 2025

@author: pc
"""
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from nilearn import plotting,image
import torch
import torch.nn.functional as F
import torch.nn as nn
import distributed as dist_fn
import seaborn as sns
from scipy import stats
from torch.utils.data import Dataset
from scipy.stats import pearsonr,kendalltau,spearmanr,ttest_rel
from sklearn.linear_model import LinearRegression

def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared, SS_Residual

def cal_cohend(x1,x2):
    diff = abs(x1-x2)
    d = np.mean(diff)/np.std(diff)
    return d

class MultiModalDataset(Dataset):

    def __init__(self, modalities):
        """
        Args:
            modalities: list of numpy arrays, each shape [N, D_i]
        """
        assert all(isinstance(m, np.ndarray) for m in modalities), "Each modality must be numpy array"
        n_samples = modalities[0].shape[0]
        assert all(m.shape[0] == n_samples for m in modalities), "All modalities must have same sample count"

        self.modalities = modalities

    def __len__(self):
        return self.modalities[0].shape[0]

    def __getitem__(self, idx):
        sample = [torch.tensor(m[idx], dtype=torch.float32) for m in self.modalities]
        return sample

class VectorQuantizerEMA(nn.Module):
    #https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=eXOBnh5pW-38
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.9, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        torch.manual_seed(0)
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
       
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        #avg_probs = torch.mean(encodings, dim=0)
        #perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized,loss, encodings

# =============== Encoder/Decoder ===============
class Encoder(nn.Module):
    def __init__(self, in_dim=232, hidden_dims=[116,58], latent_dim=18):
        super(Encoder,self).__init__()
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim,hidden_dim),
                           nn.BatchNorm1d(hidden_dim),
                           nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)
       
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_dim=232, hidden_dims=[58,116], latent_dim=18):
        super(Decoder,self).__init__()
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim,hidden_dim),
                           nn.BatchNorm1d(hidden_dim),
                           nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.net(z)


# =============== Multi-modal VQ-VAE ===============
class MultiModalVQVAE(nn.Module):
    def __init__(self, latent_shared=9, latent_private=9,
                 num_embeddings_shared=18, num_embeddings_private=18, beta=0.25):
        super().__init__()
        self.num_modalities = 3
        self.shared_vq = VectorQuantizerEMA(num_embeddings_shared, latent_shared, beta)
        self.private_vq = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings_private, latent_private, beta)
            for _ in range(self.num_modalities)
        ])

        self.encoders = nn.ModuleList([
            Encoder()
            for _ in range(self.num_modalities)
        ])
      
        self.decoders = nn.ModuleList([
            Decoder()
            for _ in range(self.num_modalities)
        ])
   
           
    

    def forward(self, xs):
        """
        xs: list of [B, D] (3 modalities)
        """
        shared_zs, private_zs, losses = [], [], []
             
        for i, x in enumerate(xs):
            
            latent = self.encoders[i](x)
           
            z_shared, z_private = latent[:, :latent.shape[1]//2], latent[:, latent.shape[1]//2:]

            # quantize
            zq_shared, vq_loss_s,_ = self.shared_vq(z_shared)
            zq_private, vq_loss_p,_ = self.private_vq[i](z_private)

            #zq = torch.cat([zq_shared, zq_private], dim=1)
            #recon = self.decoders[i](zq)
            #recons.append(recon)

            shared_zs.append(zq_shared)
            private_zs.append(zq_private)
            
            losses.append(vq_loss_s + vq_loss_p)
            
        
        self_recons = []
        for m in range(self.num_modalities):
            recon_modal = []
            for n in range(self.num_modalities):
                zq = torch.cat([shared_zs[n],private_zs[m]],dim=1)
                recon = self.decoders[m](zq)
                recon_modal.append(recon)
            self_recons.append(recon_modal)
       
        return self_recons, shared_zs, private_zs, losses


def cosine_variability(X):
    X_norm = F.normalize(X, p=2, dim=1)
    sim_matrix = X_norm @ X_norm.T
    variability = 1.0 - sim_matrix
    n = X.shape[0]
    triu_idx = torch.triu_indices(n, n, offset=1)
    sims = variability[triu_idx[0], triu_idx[1]]
    return variability.detach().numpy(),sims.mean(),sims.std()

def plot_glass_brain(atlas_path,stats_data,title,save_path,nii_path):
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    atlas_indices = np.unique(atlas_data)
    atlas_indices = np.delete(atlas_indices,0)
    stat_map = np.zeros_like(atlas_data)
    for i, idx in enumerate(atlas_indices):
        stat_map[atlas_data == int(idx)] = stats_data[i]
    stat_img = image.new_img_like(atlas_img, stat_map)
    stat_img.to_filename(nii_path)
    plt.rcParams["font.family"] = "Arial"
    display = plotting.plot_glass_brain(
        stat_img,
        title = title,
        cmap='Spectral_r',
        display_mode='lzr',
        vmin = np.min(stats_data),
        vmax = np.max(stats_data),
        colorbar=True,
        cbar_tick_format = '%.2g',
        plot_abs=False
    )
    # 修改 colorbar 刻度
    ticks = np.arange(np.min(stats_data), np.max(stats_data),1.5)  
    tick_labels = [f"{t:.2f}" for t in ticks]  
    display._colorbar_ax.get_yaxis().set_ticks(ticks)
    display._colorbar_ax.get_yaxis().set_ticklabels([str(t) for t in tick_labels])
    plt.savefig(save_path,dpi = 300)
    plt.show()

def compute_losses(xs, recons_all, vq_losses, shared_zs, private_zs,lamda1=1,lamda2 = 0,device = 'cpu'):
    
    num_modality = len(xs)
    #self_recons = [recons_all[0][0],recons_all[1][1],recons_all[2][2]]
    #reconstruction loss
    
    rec_loss = torch.tensor(0.0, device=device)

    for i in range(num_modality):
        for j in range(num_modality):
            if i==j:
            
                self_recons = recons_all[i][j]
                rec_loss = rec_loss + F.mse_loss(self_recons, xs[i],reduction='mean')
    
    
    # vq losses
    vq_loss = torch.tensor(0.0, device=device)
    for v in vq_losses:
        vq_loss = vq_loss + v.to(device)
    
    #common-specific loss
    common_loss = torch.tensor(0.0, device=device)
    for i in range(num_modality):
        for j in range(i+1,num_modality):
            common_loss = common_loss + F.mse_loss(shared_zs[i],shared_zs[j],reduction='mean')
    specific_loss = torch.tensor(0.0, device=device)
    for i in range(num_modality):
        for j in range(i+1,num_modality):
            specific_loss = specific_loss + F.mse_loss(private_zs[i],private_zs[j],reduction='mean')
    c_s_loss = common_loss / specific_loss
    
    #batch similarity losss
    
    pr_dim = private_zs[0].shape[1]
    all_privates =  torch.empty(private_zs[0].shape[0],pr_dim*3,device=device)
    for i in range(num_modality):
        start = pr_dim*i
        all_privates[:,start:start+9] = private_zs[i]
        
    all_commons = torch.zeros(shared_zs[0].shape[0], shared_zs[0].shape[1],device=device)
    for i in range(num_modality):
        all_commons =  all_commons + shared_zs[i].to(device)
    all_commons = all_commons/3
    #all_latents = torch.cat((all_commons, all_privates), dim = 1)
    
    #dist_matrix = torch.cdist(all_latents, all_latents, p=2)
    #mean_dist = dist_matrix[torch.triu(torch.ones_like(dist_matrix), diagonal=1) == 1].mean()
    #similar_loss = 1 / (1e-5 + mean_dist)
    
    #_,similar_loss = cosine_variability(all_latents)
  
    total_loss = rec_loss + vq_loss + lamda1*c_s_loss
    terms = {'recon_loss':rec_loss.item(),
             'vq_loss':vq_loss.item(),
             'c_s_loss':c_s_loss.item()}
             #'similar_loss':similar_loss.item()}
    
    return total_loss,terms

def plot_training_history(train_losses, val_losses,
                          train_recon_loss, val_recon_loss,
                          train_vq_loss, val_vq_loss,
                          train_c_s_loss, val_c_s_loss):
  
    xranges = np.arange(5,1006,200)
    sns.set_theme(style="white", palette="deep",font_scale=1.5,font="Arial")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Arial']
    
    axes[0].plot(train_losses, label='train loss', color='blue')
    axes[0].plot(val_losses, label='valid loss', color='red')
    axes[0].set_title('Total loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticks(xranges)
        
    axes[1].plot(train_recon_loss, label='train loss', color='blue')
    axes[1].plot(val_recon_loss, label='valid loss', color='red')
    axes[1].set_title('Recon loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(None)
    axes[1].set_xticks(xranges)
 
    axes[2].plot(val_vq_loss, label='valid loss', color='red')
    axes[2].set_title('VQ loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel(None)
    axes[2].set_xticks(xranges)
  
  
    axes[3].plot(train_c_s_loss, label='train loss', color='blue')
    axes[3].plot(val_c_s_loss, label='valid loss', color='red')
    axes[3].set_title('C_S loss')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel(None)
    axes[3].set_xticks(xranges)



def plot_radors(arr1,arr2,arr3,labels,yticks,save_path=None):
    rador_label = labels + [labels[0]]
    angles=np.linspace(0, 2*np.pi,len(rador_label)-1, endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))
    y_ticks_label = [str(i) for i in yticks]
    
    pcc_GM_vq = arr1[0,:].tolist()
    pcc_GM_rand = arr2[0,:].tolist()
    pcc_GM_DMVAE = arr3[0,:].tolist()
    pcc_GM_vq += pcc_GM_vq[:1]
    pcc_GM_rand += pcc_GM_rand[:1]
    pcc_GM_DMVAE += pcc_GM_DMVAE[:1]

    pcc_ALFF_vq = arr1[1,:].tolist()
    pcc_ALFF_rand = arr2[1,:].tolist()
    pcc_ALFF_DMVAE = arr3[0,:].tolist()
    pcc_ALFF_vq += pcc_ALFF_vq[:1]
    pcc_ALFF_rand += pcc_ALFF_rand[:1]
    pcc_ALFF_DMVAE += pcc_ALFF_DMVAE[:1]

    pcc_DC_vq = arr1[2,:].tolist()
    pcc_DC_rand = arr2[2,:].tolist()
    pcc_DC_DMVAE = arr3[0,:].tolist()
    pcc_DC_vq += pcc_DC_vq[:1]
    pcc_DC_rand += pcc_DC_rand[:1]
    pcc_DC_DMVAE += pcc_DC_DMVAE[:1]
    
    plt.rcParams['font.family']="Arial"
    plt.rcParams["font.size"]=20

    #colors = ['#1f77b4','#FD5109']
    colors = ['#C0D4F5','#F2CBB7','#EE8468']
    fig,ax = plt.subplots(1,3, subplot_kw={'projection': 'polar'},figsize=(30, 8))
    ax[0].plot(angles, pcc_GM_rand, '-', linewidth=4,color = colors[1])
    #ax[0].fill(angles, pcc_GM_rand, alpha=0.5,color = colors[0])
     
    ax[0].plot(angles, pcc_GM_vq , '-', linewidth=4,color = colors[0])
    #ax[0].fill(angles, pcc_GM_vq , alpha=0.2,color = colors[1])
    ax[0].plot(angles, pcc_GM_DMVAE , '-', linewidth=4,color = colors[2])
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(y_ticks_label,fontsize=18, color='gray')
    ax[0].set_thetagrids(angles * 180/np.pi, rador_label)
    ax[0].grid(color='gray', linestyle='--', linewidth=1, alpha=0.7)

    y2 = [1.0,2.0,3.0,4.0,5.0]
    y2_label = [str(i) for i in y2]
    ax[1].plot(angles, pcc_ALFF_rand, '-', linewidth=4,color = colors[1])
    #ax[1].fill(angles, pcc_ALFF_rand, alpha=0.5,color = colors[0])
     
    ax[1].plot(angles, pcc_ALFF_vq , '-', linewidth=4,color = colors[0])
    #ax[1].fill(angles, pcc_ALFF_vq , alpha=0.2,color = colors[1])
    ax[1].plot(angles, pcc_ALFF_DMVAE , '-', linewidth=4,color = colors[2])

    ax[1].set_yticks(y2)
    ax[1].set_yticklabels(y2_label,fontsize=18, color='gray')
    ax[1].set_thetagrids(angles * 180/np.pi, rador_label)
    ax[1].grid(color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax[2].plot(angles, pcc_DC_rand, '-', linewidth=4,color = colors[1])
    #ax[2].fill(angles, pcc_DC_rand, alpha=0.5,color = colors[0])
     
    ax[2].plot(angles, pcc_DC_vq , '-', linewidth=4,color = colors[0])
    #ax[2].fill(angles, pcc_DC_vq , alpha=0.2,color = colors[1])
    ax[2].plot(angles, pcc_DC_DMVAE , '-', linewidth=4,color = colors[2])

    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels(y_ticks_label,fontsize=18, color='gray')
    ax[2].set_thetagrids(angles * 180/np.pi, rador_label)
    ax[2].grid(color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.savefig(save_path,dpi=300,bbox_inches = 'tight')

def plot_points(arr1,arr2,labels,levels,save_path=None):
    #arr1[random],arr2[model]: modality*subjects*nets:numpy.array
    no_sub,no_net = arr1.shape[1],arr1.shape[2]
    
    f, ax = plt.subplots(1,3,figsize=(20, 8))
    for i in range(arr1.shape[0]):
        concat_data = np.concatenate((arr1[i,:,:].reshape(-1,1),arr2[i,:,:].reshape(-1,1)),axis = 0)
        df_concat = pd.DataFrame({'Net':labels*no_sub*len(levels),
                                  'Value':concat_data.squeeze(),
                                  'Level':[levels[0]]*no_sub*no_net+[levels[1]]*no_sub*no_net})
        sns.despine(top=True, right=True,left=True,ax = ax[i])
        sns.stripplot(df_concat,x = 'Net',y='Value',hue='Level', dodge=True, alpha=.2, legend=False,ax = ax[i])
        sns.pointplot(df_concat, x='Net', y='Value', hue='Level',
            dodge=.8 - .8 / 2, palette="dark",
            markers="*", markersize=12, linestyle="none",ax=ax[i])
        ax[i].set_yticks([-0.2,0.0,0.2, 0.4, 0.6, 0.8,1.0])
        #ax[i].set_yticklabels(['0.0','0.2', '0.4', '0.6','0.8','1.0'])
        ax[i].set(xlabel=None)
        ax[i].set(ylabel=None)
        ax[i].tick_params(axis='x', rotation=30)
        # Improve the legend
        sns.move_legend(
            ax[i], loc="lower right", ncol=2, title = None,frameon=False, columnspacing=1, handletextpad=0
        )
    #plt.show()
    f.savefig(save_path, dpi=300,bbox_inches = 'tight')
    
def paired_t_scatter(x, y,
                     labels=('Before', 'After'),
                     palette=('tab:green', 'tab:orange'),
                     figsize=(6,6),
                     jitter=0,
                     show_CI=True,
                     point_alpha=0.9,
                     save_path=None):
   
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

  
    tstat, pval = stats.wilcoxon(y, x)


    def p_to_stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'n.s.'

    stars = p_to_stars(pval)


    sns.set(style="whitegrid", font_scale=3)
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(6,6))

    x_pos = np.zeros(n)
    y_pos = np.ones(n)

    rng = np.random.default_rng(0)
    x_jitter = x_pos + rng.uniform(-jitter, jitter, size=n)
    y_jitter = y_pos + rng.uniform(-jitter, jitter, size=n)

    diffs = y - x
    up_mask = diffs >= 0
    down_mask = diffs < 0

    for i in np.where(down_mask)[0]:
        ax.plot([x_jitter[i], y_jitter[i]], [x[i], y[i]],
                color='gray', linewidth=0.8, alpha=0.6, zorder=1)

    for i in np.where(up_mask)[0]:
        ax.plot([x_jitter[i], y_jitter[i]], [x[i], y[i]],
                color='black', linewidth=0.8, alpha=0.6, zorder=1)

    ax.scatter(x_jitter, x, s=70, marker='o',
               color=palette[0], edgecolor='k', linewidth=1, alpha=point_alpha, zorder=1, label=labels[0])
    ax.scatter(y_jitter, y, s=70, marker='o',
               color=palette[1], edgecolor='k', linewidth=1, alpha=point_alpha, zorder=1, label=labels[1])

    def mean_ci(arr):
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        m = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        ci = 1.96 * se
        return m, m-ci, m+ci

    if show_CI:
        m1, lo1, hi1 = mean_ci(x)
        m2, lo2, hi2 = mean_ci(y)
        ax.errorbar(-0.08, m1, yerr=[[m1-lo1],[hi1-m1]], fmt='s', color='k',
                    elinewidth=1.0, capsize=6, capthick=1.0, markerfacecolor=palette[0], markersize=8, zorder=4)
        ax.errorbar(1.08, m2, yerr=[[m2-lo2],[hi2-m2]], fmt='s', color='k',
                    elinewidth=1.0, capsize=6, capthick=1.0, markerfacecolor=palette[1], markersize=8, zorder=4)

        ax.hlines(m1, -0.2, 0.2, colors=palette[0], linewidth=2, alpha=0.9, zorder=3)
        ax.hlines(m2, 0.8, 1.2, colors=palette[1], linewidth=2, alpha=0.9, zorder=3)

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0,1])
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('PCC', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.legend().remove()
    
    ymax = np.nanmax(np.concatenate([x,y]))
    ytext = ymax + 0.07*(np.nanmax(np.concatenate([x,y])) - np.nanmin(np.concatenate([x,y])))
    ax.text(0.5, ytext, f'Wilcoxon signed-rank test : w={tstat:.2f}, p={pval:.3f} ({stars})',
            ha='center', va='bottom', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300,bbox_inches = 'tight')
    return tstat, pval

def plot_combined_heatmap(mat1,mat2,save_path):
    C = np.zeros_like(mat1)
    n = C.shape[0]


    C[np.triu_indices_from(C, k=1)] = mat1[np.triu_indices_from(mat1, k=1)]


    C[np.tril_indices_from(C, k=-1)] = mat2[np.tril_indices_from(mat2, k=-1)]

    np.fill_diagonal(C, np.nan)

    sns.set_theme(style="white", palette="deep",font_scale=1.2,font="Arial")

    mask_upper = np.tril(np.ones_like(C, dtype=bool), k=-1) 
    mask_lower = np.triu(np.ones_like(C, dtype=bool), k=1)   


    cmapA = matplotlib.colormaps.get_cmap("Oranges")    
    cmapB = matplotlib.colormaps.get_cmap("Blues")    

    ax=sns.heatmap(C, mask=mask_upper, cmap=cmapA, square=True,
                cbar=True, vmin=0, vmax=np.nanmax(mat1))

    ax=sns.heatmap(C, mask=mask_lower, cmap=cmapB, square=True,
                cbar=True, vmin=0, vmax=np.nanmax(mat2))

    ax.plot(
        np.arange(n) + 0.5,
        np.arange(n) + 0.5,
        linestyle='-',
        color='black',
        linewidth=1,
        alpha=1
    )
   
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(axis='both', which='both', length=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")  
    plt.savefig(save_path, dpi=300)

def plot_paired_expression(x1,y1,x2,y2,save_path):
    # --- Color Design ---
    warm_color = "#E6863B"   # warm orange
    cool_color = "#4C72B0"   # cool blue

    # --- Seaborn Style ---
    sns.set_theme(style="white", font="Arial", font_scale=1.8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ======== PLOT 1: Warm Tone ========
    sns.regplot(
        x=x1, y=y1,
        ax=axes[0],
        scatter_kws={"s": 80, "color": warm_color, "alpha": 0.4},
        line_kws={"color": warm_color, "linewidth": 2}
    )
    axes[0].set(xlabel = 'Gene expression (PLS1 score)')
    axes[0].set(ylabel = 'Private latent expression')

    # Correlation annotation
    r1, p1 = pearsonr(x1, y1)
    axes[0].text(
        0.05, 0.92, f"r = {r1:.2f}\np = {p1:.3f}",
        transform=axes[0].transAxes,
        fontsize=14,
        color=warm_color,
        verticalalignment="top"
    )

    # ======== PLOT 2: Cool Tone ========
    sns.regplot(
        x=x2, y=y2,
        ax=axes[1],
        scatter_kws={"s": 80, "color": cool_color, "alpha": 0.4},
        line_kws={"color": cool_color, "linewidth": 2}
    )
    axes[1].set(xlabel = 'Gene expression (PLS1 score)')
    axes[1].set(ylabel = 'Shared latent expression')

    r2, p2 = pearsonr(x2, y2)
    axes[1].text(
        0.05, 0.92, f"r = {r2:.2f}\np = {p2:.3f}",
        transform=axes[1].transAxes,
        fontsize=14,
        color=cool_color,
        verticalalignment="top"
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_paired_bar(bar1,bar2,save_path):
    df1 = pd.DataFrame({'label':['PC'+str(i) for i in range(1,6)],'Value':bar1})
    df2 = pd.DataFrame({'label':['PC'+str(i) for i in range(1,6)],'Value':bar2})

    sns.set_theme(style="whitegrid", font="Arial", font_scale=1.8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data = df1,ax=axes[0],
                x = 'Value',y='label',width=0.5,
                orient='h',hue = 'Value',legend = None,edgecolor="black", linewidth=1.2,palette='Oranges')
    sns.despine(left=False, bottom=False)
    axes[0].set(xlabel='Correlation')
    axes[0].set(ylabel=None)
    axes[0].set_xlim(-0.5,0.5)
    
    sns.barplot(data = df2,ax=axes[1],
                x = 'Value',y='label',width=0.5,
                orient='h',hue = 'Value',legend = None,edgecolor="black", linewidth=1.2,palette='Blues')
    sns.despine(left=False, bottom=False)
    axes[1].set(xlabel='Correlation')
    axes[1].set(ylabel=None)
    axes[1].set_xlim(-0.5,0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def cal_representation_similarity(latent,df,label,category):
    phenotye = df[label]
   
    person_v = phenotye.dropna()
    idx = person_v.index.to_list()
    latent = latent[idx,:]
    no_subs = len(person_v)
    individual_sim = np.zeros((no_subs,no_subs))
    
    for i in range(no_subs):
        for j in range(no_subs):
            if j>i:
                if category == 'C':
                    individual_sim[i,j] = abs(person_v[i] - person_v[j])
                else:
                    individual_sim[i,j] = int(person_v[i] == person_v[j])
                    
    individual_sim = individual_sim + individual_sim.T
    
    RDM,_ = cosine_variability(latent)
    triu_idx = torch.triu_indices(RDM.shape[0], RDM.shape[0], offset=1)
    var_RDM = RDM[triu_idx[0], triu_idx[1]]
    var_sim = individual_sim[triu_idx[0], triu_idx[1]]
    [r,p] = kendalltau(var_RDM,var_sim)
    
    return r,p,var_RDM,individual_sim

def plot_r_diffs(r_diffs:list,savepath):
    groups = {
    "GMV": r_diffs[0],
    "ALFF": r_diffs[1],
    "FDC": r_diffs[2]
}

    colors = {
        "GMV": "#E69F00",   # Blue
        "ALFF": "#0072B2",   # Orange
        "FDC": "#009E73"    # Green
    }
    
    sns.set_theme('paper','whitegrid',font = 'Arial',font_scale=2.2)
    fig, axes = plt.subplots(len(groups), 1, figsize=(6, 4), sharex=True)
    
    if len(groups) == 1:
        axes = [axes]  
    
    for ax, (name, data) in zip(axes, groups.items()):
    
        # KDE
        sns.kdeplot(
            data,
            ax=ax,
            color=colors[name],
            linewidth=2,
            fill=True,
            alpha=0.25
        )
    
        mean_val = np.mean(data)
        ax.axvline(mean_val, ymin=0, ymax=0.42,   
                   color="black", linewidth=2)
    
        ax.set_ylabel(name, rotation=0, labelpad=20, fontsize=12)
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.spines["left"].set_visible(False)
    
    axes[-1].set_xlabel("Value", fontsize=12)
    sns.despine(left=True)
    plt.savefig(savepath, dpi=300,bbox_inches='tight')

def plot_3D_bar(TPR,FPR,SR,save_path):
    x = np.array(TPR)
    y = np.array(FPR)
    z = np.array(SR)

    colors = [  
              "#5E3C99",  # deep purple-blue,w\o tuning
              "#1B9E77",  # teal green, fully fine tuning
              "#E66101"]  # warm orange, partial fine tuning

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # bar size
    dx = dy = 0.05
    # plot 3 bars
    for i in range(3):
        ax.bar3d(
            x[i], y[i], 0,     # base
            dx, dy, z[i],      # width, depth, height
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.3,
            shade=True
        )

    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_zlabel(None)


    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.view_init(elev=20, azim=-60)   
    ax.dist = 0                     
    plt.savefig(save_path, dpi=300, bbox_inches = 'tight')

def cal_spatial_neuromap(private,shared,original,neuro_par):
    pr_corr = []
    pr_R2 = []
    for pr in private:
        sub_pr = pr.cpu().numpy()
        spatial_r = [pearsonr(sub_pr[i,:],neuro_par.squeeze())[0] for i in range(sub_pr.shape[0])]
        R2 = [get_reg_r_sq(neuro_par.reshape(-1,1), sub_pr[i,:].reshape(-1,1))[0][0] for i in range(sub_pr.shape[0])]
        pr_corr.append(spatial_r)
        pr_R2.append(R2)
    
    sh_corr = []
    sh_R2 = []
    for pr in shared:
        sub_pr = pr.cpu().numpy()
        spatial_r = [pearsonr(sub_pr[i,:],neuro_par.squeeze())[0] for i in range(sub_pr.shape[0])]
        R2 = [get_reg_r_sq(neuro_par.reshape(-1,1), sub_pr[i,:].reshape(-1,1))[0][0] for i in range(sub_pr.shape[0])]
        sh_corr.append(spatial_r)
        sh_R2.append(R2)

    real_corr = []
    real_R2 = []
    for pr in original:
        sub_pr = pr.cpu().numpy()
        spatial_r = [pearsonr(sub_pr[i,:],neuro_par.squeeze())[0] for i in range(sub_pr.shape[0])]
        R2 = [get_reg_r_sq(neuro_par.reshape(-1,1), sub_pr[i,:].reshape(-1,1))[0][0] for i in range(sub_pr.shape[0])]
        real_corr.append(spatial_r)
        real_R2.append(R2)
    print(f'Private vs real:\n{ttest_rel(np.abs(pr_corr[0]),np.abs(real_corr[0]))},{ttest_rel(np.abs(pr_corr[1]),np.abs(real_corr[1])),{ttest_rel(np.abs(pr_corr[2]),np.abs(real_corr[2]))}}')
    print(f'Shared vs real:\n{ttest_rel(np.abs(sh_corr[0]),np.abs(real_corr[0]))},{ttest_rel(np.abs(sh_corr[1]),np.abs(real_corr[1])),{ttest_rel(np.abs(sh_corr[2]),np.abs(real_corr[2]))}}')
    print(f'Private vs real:\n{ttest_rel(pr_R2[0],real_R2[0])},{ttest_rel(pr_R2[1],real_R2[1])},{ttest_rel(pr_R2[2],real_R2[2])}')
    print(f'Shared vs real:\n{ttest_rel(sh_R2[0],real_R2[0])},{ttest_rel(sh_R2[1],real_R2[1])},{ttest_rel(sh_R2[2],real_R2[2])}')
        
    return pr_corr,real_corr,sh_corr,pr_R2,real_R2,sh_R2

def cal_multi_regress(private,shared,original,neuro_par):
    
    pr_R2 = []
    for pr in private:
        sub_pr = pr.cpu().numpy()
        R2 = [get_reg_r_sq(neuro_par, sub_pr[i,:].reshape(-1,1))[0][0] for i in range(sub_pr.shape[0])]
        pr_R2.append(R2)
        
    sh_R2 = []
    for pr in shared:
        sub_pr = pr.cpu().numpy()
        R2 = [get_reg_r_sq(neuro_par, sub_pr[i,:].reshape(-1,1))[0][0] for i in range(sub_pr.shape[0])]
        sh_R2.append(R2)
    
    real_R2 = []
    for pr in original:
        sub_pr = pr.cpu().numpy()
        R2 = [get_reg_r_sq(neuro_par, sub_pr[i,:].reshape(-1,1))[0][0] for i in range(sub_pr.shape[0])]
        real_R2.append(R2)
    
    print(f'Private vs real:\n{ttest_rel(pr_R2[0],real_R2[0])},{ttest_rel(pr_R2[1],real_R2[1])},{ttest_rel(pr_R2[2],real_R2[2])}')
    print(f'Shared vs real:\n{ttest_rel(sh_R2[0],real_R2[0])},{ttest_rel(sh_R2[1],real_R2[1])},{ttest_rel(sh_R2[2],real_R2[2])}')
       
    
    return pr_R2,real_R2,sh_R2
    

