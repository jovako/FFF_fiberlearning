import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

def plot_mnist(xs, titles, n=8, mark_first=False):
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=len(xs), ncols=1)
    for j, subfig in enumerate(subfigs):
        subfig.suptitle(f"{titles[j]}")
        axes = subfig.subplots(nrows=1, ncols=n)
        for i in range(n):
            axes[i].imshow(xs[j][i].detach().cpu().reshape(16, 16).T, cmap='gray', vmin=0, vmax=1)
            axes[i].xaxis.set_tick_params(labelbottom=False)
            axes[i].yaxis.set_tick_params(labelleft=False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        if mark_first:
            for spine in axes[0].spines.values():
                spine.set_edgecolor('red')
            spine.set_linewidth(2)
    return fig

def plot_sm(model, i_sample, x, x_orig, i_plot, n, name="learned"):
    subject_model = model.subject_model
    plot_dict = {"ind": i_sample, "n": n+1, "mark_first": r"f(x)"}
    conditioned = subject_model.apply_conditions((x,))
    c_sm = conditioned.condition
    x_sm = conditioned.x0
    xc = subject_model.encode(x_sm.to(subject_model.device), c_sm.to(subject_model.device))
    xcx = subject_model.decode(xc, c_sm.to(subject_model.device))
    xc = (xc.detach().cpu() - model.data_shift) / model.data_scale

    # plotting reconstructions
    plot_dict["x_plot"] = torch.cat((x_orig.detach().cpu().unsqueeze(0), xcx.detach().cpu()[i_plot]), dim=0)
    #x_plot = fiber_rec[i_plot].detach().cpu()
    if name=="learned":
        plot_dict["save_name"] = "1v_x_px_x"
        plot_dict["suptitle"] = "Fiber samples forwarded by the subject model $\quad f(D(t^\dagger(v)))$"
    elif name=="PGD":
        plot_dict["save_name"] = "3v_x_c_x"
        plot_dict["suptitle"] = "Corrected fiber samples forwarded by the subject model $\quad f(\mathrm{GD}(D(t^\dagger(v))))$"
    elif name=="NNs":
        plot_dict["save_name"] = "5NN_c_x"
        plot_dict["suptitle"] = "Reconstruct NNs with subject model $\quad f(\mathrm{NN})$"
    
    return xc, plot_dict

def plot_PGD(model, latent_dim, i_sample, z, x, x0, c0, n, n_steps=100):
    plot_dict = {"ind": i_sample, "mark_first": r"original $x$", "save_name": "2v_x_px", "n": n}
    px = []
    res = []
    px.append(x0.unsqueeze(dim=0))
    for zi, xi in zip(z,x):
        pxi, resi = calc_PGD(model, latent_dim, zi, xi, x0, c0, steps=n_steps)
        px.append(pxi)
        res.append(resi)

    px = torch.cat(px,dim=0)
    res = torch.cat(res,dim=0).detach().cpu()
    plot_dict["suptitle"] = "Gradient descent on learned fiber samples $\quad \mathrm{GD}(D(t^\dagger(v)))$"
    plot_dict["x_plot"] = px.clone()
    plot_dict_res = plot_dict.copy()
    plot_dict_res["x_plot"] = px.clone()
    plot_dict_res["save_name"] = "2v_x_px_res"
    plot_dict_res["x_plot"][1:] = torch.abs(plot_dict["x_plot"][1:] - x)
    plot_dict_res["suptitle"] = "Residual between learned and corrected fiber $\quad |D(t^\dagger(v))-\mathrm{GD}(D(t^\dagger(v)))|$"
    #print(torch.max(plot_dict["x_plot"][1:],dim=1).values)
    return px[1:], res, plot_dict, plot_dict_res

def calc_NNs(latent_dim, verify, train_c, train_samples, n=8):
    # calculate Nearest Neighbours
    diff = train_c - verify
    diff = torch.sqrt(torch.sum(torch.square(diff), dim =1)/latent_dim)
    ind = torch.argsort(diff)[:n]
    diff = diff[ind]
    #print("squared distance between chosen sample and nearest train samples:", squared_diff[ind]/5)
    similar = train_samples[ind]
    #labels = train_l[ind].numpy()
    labels = torch.zeros(1000000)[ind].numpy()
    NN = diff[0].numpy()
    diff_round = np.round(diff.numpy(),2)
    labels = np.concatenate((np.array([8]), labels))
    diff_round = np.concatenate((np.array([8]), diff_round))

    # plot NNs
    suptitle = r"Nearest Neighbours in data with distance $\sqrt{\frac{(c_{orig}-c_i)^2}{dim_c}}$"
    #def title_func(label, distance):
    #    return f"lab: {label}, dist: {distance:.2f}"
    titles = (labels, diff_round)
    #plot_images(i_sample, similar, titles=titles, suptitle=suptitle)
    
    #dist = torch.sqrt(torch.sum((similar[:-1]-similar[1:])**2,-1))
    #print(dist)
    return NN, similar, suptitle, titles

def plot_images(ind, x_plot, n=8, titles=None, suptitle=None, mark_first=False, mark_second=False, save=False, save_name=None):
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(15, 3))
    for i in range(n):
        axes[i].imshow(x_plot[i].detach().cpu().reshape(16, 16).T, cmap='gray', vmin=0, vmax=1)
        axes[i].xaxis.set_tick_params(labelbottom=False)
        axes[i].yaxis.set_tick_params(labelleft=False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        if isinstance(titles, np.ndarray):
            axes[i].set_title(titles[i])
        elif isinstance(titles, tuple) and i>0:
            axes[i].set_title(f"label: {titles[0][i]}")
            axes[i].set_xlabel(f"dist: {titles[1][i]:.2f}")
        elif titles != None:
            axes[i].set_title(titles[i])
    if mark_first:
        for spine in axes[0].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
        axes[0].set_title(mark_first)
    if mark_second:
        for spine in axes[1].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2)
        axes[1].set_xlabel(r"VAE($x$)")
    if suptitle != None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    path = f"plots/{plot_dir}/{nums[ind]}_{save_name}.png"
    if save:
        plt.savefig(path, bbox_inches='tight')
        #plt.close()
    return path

def plot_fiber_check(zrange, latent_dim, check_fiber, c_sample, NN, title, i_plot, delta2, N, upperbound=True, save=False):
    
    delta_coarse = torch.sqrt(torch.sum((check_fiber-c_sample.repeat(40,1))**2, dim=1)/latent_dim)
    print(torch.max(delta_coarse.reshape(40,-1)))
    delta_c_std, delta_c = torch.std_mean(delta_coarse.reshape(40,-1), dim=0)
    delta_c_std_1, delta_c_1 = torch.std_mean(delta_coarse.reshape(40,-1)[1:,:], dim=0)
    delta_c_std[1], delta_c[1]
    delta_c = delta_c.numpy()
    delta_details = np.linspace(-zrange,zrange,N)
    
    #plotting...
    plt.figure(figsize=[10,5])
    plt.errorbar(delta_details,delta_c,yerr=delta_c_std, label="average deviation",capsize=5)
    plt.plot(delta_details[i_plot],delta_coarse[i_plot], marker='o', label="shown samples", linestyle='')
    if delta2!=None:
        plt.plot(delta_details[i_plot],delta2[:i_plot.shape[0]], marker='o', label="shown corr. samples", linestyle='')
    
    plt.axhline(y=NN, xmin=-10, xmax=10, color='gray', linestyle="--", label='nearest train sample')
    if upperbound==True:
        plt.axhline(y=1.49, xmin=-10, xmax=10, color='red', linestyle="--", label='average random')
    plt.title(title)
    plt.ylabel(r"$\sqrt{\frac{(c_{true}-c_{rec})^2}{dim_c}}$", fontsize=22)
    plt.legend(bbox_to_anchor=(0.98, 0.5), borderaxespad=1)
    plt.xlabel(r"latent dim 0 [$\sigma$]")
    plt.yscale("log")
    #plt.ylim(0, 1.6)
    plt.xlim((-zrange,zrange))
    plt.grid()
    if save:
        plt.savefig(f"plots/{plot_dir}/{nums[j]}deviations.png", bbox_inches='tight')

def calc_PGD(model, latent_dim, z, x, x0, c0, steps=100):
    subject_model = model.subject_model
    torch.set_grad_enabled(True)
    z = torch.unsqueeze(z,dim=0)
    constraint = torch.sqrt(torch.sum(((x0 - x)**2), dim=-1))
    #print("distance: x_original - x_fiber: ", constraint.item())
    c0 = torch.unsqueeze(c0,dim=0)
    
    def closs(c0, c_hat, x_hat):
        v = torch.sqrt(torch.sum(((c0 - c_hat)**2), dim=-1))/max(latent_dim, 1)
        #distance = torch.sqrt(torch.sum(((x_hat.detach().cpu() - vx[0])**2), dim=-1))
        #r = torch.max(torch.Tensor([0., (distance - constraint).item()]))
        #print("loss:", v.item())
        #if r != 0:
        #    print("penalty:", r.item())
        return v
    
    class ZModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(z.clone())
    
    zmodel = ZModel()
    optimizer = torch.optim.Adam(zmodel.parameters(), lr=0.05)
    max_z = torch.max(torch.abs(zmodel.param))
    
    c_empty = torch.empty((0,), device=subject_model.device, dtype=x.dtype)
    for i in range(steps):
        zx = model.decode(zmodel.param.to(subject_model.device), c_empty)
        conditioned = subject_model.apply_conditions((zx,))
        c_sm = conditioned.condition
        x_sm = conditioned.x0
        zxc = (subject_model.encode(x_sm, c_sm).cpu() - model.data_shift) / model.data_scale
        loss = closs(c0, zxc, zx)
        loss.backward()
        optimizer.step()
        # check, if image gets further apart than 
        #check = torch.mean(torch.sqrt(torch.sum(((x0 - zx.cpu())**2), dim=-1)))
        #if check > constraint:
        #    print("too far:")
        #    print(check.item())
        optimizer.zero_grad()

    #print("loss:", loss.item())
    torch.set_grad_enabled(False)
    max_z_after = torch.max(torch.abs(zmodel.param))
    if max_z_after > 5. and (max_z_after-max_z) > 1.:
        print("warning!, z got an extreme value:", max_z_after)
    zx = model.decode(zmodel.param.to(subject_model.device), c_empty).detach().cpu()
    return zx, loss
