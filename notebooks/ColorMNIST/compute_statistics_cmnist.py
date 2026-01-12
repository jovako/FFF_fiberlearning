import h5py
import numpy as np
from scipy.special import kl_div, rel_entr
import torch
import lightning_trainable
import os
import fff
import matplotlib.pyplot as plt
from fff.evaluate.fid import compute_fid_openai_tf as compute_fid
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"

log_folder = "/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/color_logs"
plot_folder = "/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/col_mnist/col_mnist_plots/"

def Decolorize(x_colored):
    def detect_colors(x_data):
        background_colors = torch.mean(x_data[:,:,:,0],-1)
        return background_colors
    x_c = x_colored.reshape(-1,3,28,28)
    c = detect_colors(x_c)
    c_image = c.unsqueeze(-1).expand(-1,3,28*28).reshape(-1,3,28,28)
    x_dc = (x_c-c_image) / ((c_image+0.5)%1 - c_image)
    return x_dc.abs(), c

def normal(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi)/sigma
def gaussian_mix_dense(x):
    return 0.6 * normal(x, 0.7, 0.08) + 0.35 * normal(x, 0.5, 0.015) + 0.05 * normal(x, 0.1, 0.02)


class SubjectModelInterface:
    def __init__(self, subject_model):
        self.subject_model = subject_model

    def __call__(self, x):
        return self.subject_model.encode(x, torch.empty((x.shape[0], 1), device=x.device))

def load_model(name):
    if "ndtm" in name:
        name = "250_nf"
    try:
        checkpoint = lightning_trainable.utils.find_checkpoint(root=f"{log_folder}/{name}", version=0, epoch="best")
    except:
        checkpoint = lightning_trainable.utils.find_checkpoint(root=f"{log_folder}/{name}", version=0, epoch="last")
    ckpt = torch.load(checkpoint, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    hparams["cond_dim"] = 0
    hparams["data_set"]["root"] = '/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/cc_mnist'
    hparams["data_set"]["subject_model_path"] = '/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/cc_mnist/subject_model/checkpoints/299_fixed.ckpt'
    hparams["lossless_ae"] = {"model_spec": hparams["lossless_ae"]}
    hparams["load_lossless_ae_path"] = '/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/color_logs/Lossless_VAE/checkpoints/last_fixed.ckpt'
    if "FlowMatching" in hparams["density_model"][0]["name"]:
        hparams["density_model"][0]["interpolation_schedule"] = hparams["density_model"][0].pop("interpolation")
        hparams["density_model"][0].pop("sigma")
        hparams["density_model"][0]["network_hparams"] = {"layers_spec": hparams["density_model"][0].pop("layers_spec")}
    model = fff.fiber_model.FiberModel(hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model

def save_model_samples(samples, originals, sample_embeddings, original_embeddings, name):
    save_path = os.path.join(plot_folder, name)
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "samples": samples,
        "originals": originals,
        "sample_embeddings": sample_embeddings,
        "original_embeddings": original_embeddings
    }, os.path.join(save_path, "samples.pt"))

def save_model_stats(fl_stats, kl_stats, w1_stats, dev_stats, fid_stats, name):
    save_path = os.path.join(plot_folder, name)
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "fl_stats": fl_stats,
        "kl_stats": kl_stats,
        "w1_stats": w1_stats,
        "dev_stats": dev_stats,
        "fid_stats": fid_stats,
    }, os.path.join(save_path, "stats.pt"))
    
def load_model_samples(name):
    load_path = os.path.join(plot_folder, name, "samples.pt")
    samples_dict = torch.load(load_path)
    return samples_dict["samples"], samples_dict["originals"], samples_dict["sample_embeddings"], samples_dict["original_embeddings"], 


def load_model_stats(name):
    load_path = os.path.join(plot_folder, name, "stats.pt")
    stats_dict = torch.load(load_path, weights_only=False)
    return stats_dict["fl_stats"], stats_dict["kl_stats"], stats_dict["w1_stats"], stats_dict["dev_stats"], stats_dict["fid_stats"],

@torch.no_grad()
def evaluate_model(model_name, dataset=None, samples_per_image=10, batch_size=512, save=True):
    print(f"Evaluating model {model_name}")
    fiber_model = load_model(model_name)
    subject_model = SubjectModelInterface(fiber_model.subject_model)
    
    if dataset is not None:
        if not isinstance(dataset, torch.utils.data.TensorDataset):
            dataset = torch.utils.data.TensorDataset(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    try:
        samples, originals, sample_embeddings, original_embeddings = load_model_samples(model_name)
        if dataset is not None:
            if torch.mean((dataset[:][0] - originals)**2) > 1.e-6:
                raise ValueError("Dataset not identical to precomputed dataset")
            else:
                print("Using precomputed dataset")
    except Exception as e:
        # print(e)
        assert dataset is not None, "If no precomputed samples are available, dataset has to be passed"
    
        samples = []
        originals = []
        sample_embeddings = []
        original_embeddings = []
        
        for n_batch, batch in enumerate(dataloader):
            x = batch[0].to(device).float()
            test_image_embedding = subject_model(x)

            samples_image = []
            embeddings_image = []
            for i in range(samples_per_image):
                samples_image.append(fiber_model.sample(torch.Size([x.shape[0]]), test_image_embedding.to(device)).reshape(x.shape[0], 3, 28, 28))
                embeddings_image.append(subject_model(samples_image[-1]))
            samples.append(torch.stack(samples_image, dim=1))
            sample_embeddings.append(torch.stack(embeddings_image, dim=1))
            originals.append(x)
            original_embeddings.append(test_image_embedding)
            
        samples = torch.cat(samples, dim=0)
        sample_embeddings = torch.cat(sample_embeddings, dim=0)
        originals = torch.cat(originals, dim=0)
        original_embeddings = torch.cat(original_embeddings, dim=0)
        
        if save:
            save_model_samples(samples, originals, sample_embeddings, original_embeddings, model_name)
    
    fl_stats = compute_fiber_loss_model(originals.to(device), 
                                        samples.to(device), 
                                        original_embeddings.to(device), 
                                        sample_embeddings.to(device), 
                                        subject_model)
    kl_stats, w1_stats, dev_stats = compute_kl_w1_and_deviation(samples.permute(1, 0, 2, 3, 4))
    fid_stats = compute_fid_with_std(originals.to(device), samples.to(device))
    if save:
        save_model_stats(fl_stats, kl_stats, w1_stats, dev_stats, fid_stats, model_name)
    return fl_stats, kl_stats, w1_stats, dev_stats, fid_stats
    
@torch.no_grad()
def compute_fiber_loss_model(originals, samples, original_embeddings, sample_embeddings, subject_model, batch_size=2048):
    N=None
    n_rows=5
    fiber_loss = []
    
    dataset = torch.utils.data.TensorDataset(originals, samples, original_embeddings, sample_embeddings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Computing Fiber Loss...")
    
    for sam_i in range(samples.shape[1]):
        fiber_loss_sami = []

        for i, batch in enumerate(dataloader):
            test_samples, x_sampled, test_c, xc = batch
            x_sampled, xc = x_sampled[:,sam_i], xc[:,sam_i]
            if i == 0 and sam_i == 0:
                # sanity check
                xc_recomputed = subject_model(x_sampled)
                assert torch.sqrt(torch.mean((xc-xc_recomputed)**2,-1)).mean() < 1.e-4, f"Fiber loss between identical samples is {torch.sqrt(torch.sum((xc-xc_recomputed)**2,-1)/float(xc.shape[-1])).mean()}"
            

            fiber_loss_sami.append(torch.sqrt(torch.sum((xc-test_c)**2,-1)/float(xc.shape[-1])))
        fiber_loss.append(torch.cat(fiber_loss_sami,0).cpu())
    
    fl_mean, fl_std = torch.cat(fiber_loss, dim=0).mean(), torch.cat(fiber_loss, dim=0).std() 
    print("fiber_loss mean is: ", fl_mean, " +- ", fl_std)
    
    return (fl_mean, fl_std)

@torch.no_grad()
def compute_fid_with_std(originals, samples):
    print("Computing FID...")
    fids = []
    for sam_i in range(samples.shape[1]):
        fid = compute_fid(originals, samples[:,sam_i])[1]
        fids.append(fid)
    fid_mean, fid_std = np.mean(fids), np.std(fids)
    print("fid mean is: ", fid_mean, " +- ", fid_std)
    
    return (fid_mean, fid_std)

def compute_kl_w1_and_deviation(sample_list, n_bins=100):
    print("Computing KL Divergence...")
    kls_r, kls_g, kls_b = [], [], []
    w1s_r, w1s_g, w1s_b = [], [], []
    dev = []
    for samples in sample_list:
        x_dc, colors = Decolorize(samples)
        max_pix = torch.max(x_dc.mean(1).reshape(-1,28*28), -1)[0].cpu()
        dev.append(torch.abs(max_pix - 1).mean().numpy())

        kls_c, w1s_c = [], []
        for c in range(3):
            H, bins = np.histogram(colors[:, c].cpu(), bins=n_bins, range=[0,1], density=True)
            bin_width = bins[1] - bins[0]
            mids = bins[:-1] + bin_width / 2

            # Discretized probabilities
            p = H * bin_width
            q = gaussian_mix_dense(mids) * bin_width
            p /= p.sum()
            q /= q.sum()

            # KL divergence
            kl_per_bin = rel_entr(p, q)
            kl_per_bin = kl_per_bin[~np.logical_or(np.isnan(kl_per_bin), np.isinf(kl_per_bin))]
            kl = np.sum(kl_per_bin)
            kls_c.append(kl)

            # Wasserstein-1 distance
            C_p = np.cumsum(p)
            C_q = np.cumsum(q)
            tv_per_bin = np.abs(C_p - C_q)
            tv_per_bin = tv_per_bin[~np.logical_or(np.isnan(tv_per_bin), np.isinf(tv_per_bin))]
            w1 = np.sum(tv_per_bin) * bin_width
            w1s_c.append(w1)

        kls_r.append(kls_c[0]); kls_g.append(kls_c[1]); kls_b.append(kls_c[2])
        w1s_r.append(w1s_c[0]); w1s_g.append(w1s_c[1]); w1s_b.append(w1s_c[2])

    # Aggregate statistics
    def summarize(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean, std

    kl_means = [summarize(x) for x in [kls_r, kls_g, kls_b]]
    w1_means = [summarize(x) for x in [w1s_r, w1s_g, w1s_b]]
    dev_mean, dev_std = summarize(dev)

    print("Red KL mean is: ", kl_means[0][0], " +- ", kl_means[0][1])
    print("Green KL mean is: ", kl_means[1][0], " +- ", kl_means[1][1])
    print("Blue KL mean is: ", kl_means[2][0], " +- ", kl_means[2][1])

    return kl_means, w1_means, (dev_mean, dev_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default="all",
                    help="Which model class to evaluate")
    parser.add_argument("--include_test_data", type=bool, default=False,
                    help="Whether to provide test data")
    args = parser.parse_args()

    model_names = {
        #"fff": ["250_fff", "250_fff_0", "250_fff_1", "250_fff_2"],
        #"f3f": ["250_f3f", "250_f3f_0", "250_f3f_1", "250_f3f_2"],
        #"nf": ["250_nf", "250_nf_0", "250_nf_1", "250_nf_2"],
        #"dnf": ["100_dnf", "100_dnf_0", "100_dnf_1", "100_dnf_2"],
        #"mlf": ["100_mlf", "250_mlf_0", "250_mlf_1"],
        #"diff": ["1000_diff"],
        #"cfm": ["250_cfm"],
        "ndtm": ["250_ndtm_0", "250_ndtm_1", "250_ndtm_2", "250_ndtm_3"],
        "ndtm_correlated": ["250_ndtm_correlated_0", "250_ndtm_correlated_1", "250_ndtm_correlated_2", "250_ndtm_correlated_3"],
    }

    if args.include_test_data:
        with h5py.File('/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/cc_mnist/data.h5', 'r') as f:
            test_data = torch.from_numpy(f['test_images'][:])
        print("Recomputing statistics with test data")
    else:
        test_data = None
        print("Using precomputed statistics")
        
    if args.model_class == "all":
        for model_class in model_names.keys():
            for model_name in model_names[model_class]:
                evaluate_model(model_name, dataset=test_data)
    else:
        for model_name in model_names[args.model_class]:
            evaluate_model(model_name, dataset=test_data)
