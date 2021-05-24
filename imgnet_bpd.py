import os
from utils.torch.rand import *
from model.cifar_train import Model
from torch.utils.data import *
from discretization import *
from torchvision import datasets, transforms
import random
import time
import argparse
from tqdm import tqdm
import pickle
import mydatasets

def eval_bpd(quantbits, nz, gpu):
    # model and compression params
    zdim = 8*16*16
    zrange = torch.arange(zdim)
    xdim = 32**2 * 3
    xrange = torch.arange(xdim)
    ansbits = 31 # ANS precision
    type = torch.float64 # datatype throughout compression
    device = f"cuda:{gpu}" # gpu

    # set up the different channel dimension for different latent depths
    if nz == 8:
        reswidth = 252
    elif nz == 4:
        reswidth = 254
    elif nz == 2:
        reswidth = 255
    else:
        reswidth = 256
    assert nz > 0

    print(f"CIFAR - {nz} latent layers - {quantbits} bits quantization")

    # seed for replicating experiment and stability
    np.random.seed(100)
    random.seed(50)
    torch.manual_seed(50)
    torch.cuda.manual_seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # compression experiment params
    experiments = 1
    ndatapoints = 50000

    # <=== MODEL ===>
    model = Model(xs = (3, 32, 32), nz=nz, zchannels=8, nprocessing=4, kernel_size=3, resdepth=8, reswidth=reswidth).to(device)
    model.load_state_dict(
        torch.load(f'model/params/imagenet/nz{nz}',
                   map_location=lambda storage, location: storage
                   )
    )
    model.eval()

    print("Discretizing")
    # get discretization bins for latent variables
    zendpoints, zcentres = discretize(nz, quantbits, type, device, model, "imagenet")

    # get discretization bins for discretized logistic
    xbins = ImageBins(type, device, xdim)
    xendpoints = xbins.endpoints()
    xcentres = xbins.centres()

    print("Load data..")
    # <=== DATA ===>
    class ToInt:
        def __call__(self, pic):
            return pic * 255
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
    test_set = mydatasets.Imgnet32Val(transform=transform_ops)

    # sample (experiments, ndatapoints) from test set with replacement
    randindices = np.random.choice(len(test_set.data), size=(experiments, ndatapoints), replace=False)

    print("Setting up metrics..")
    # metrics for the results
    elbos = np.zeros((experiments, ndatapoints), dtype=np.float)

    print("Compression..")
    for ei in range(experiments):
        print(f"Experiment {ei + 1}")
        subset = Subset(test_set, randindices[ei])
        test_loader = DataLoader(
            dataset=subset,
            batch_size=1, shuffle=False, drop_last=True)
        datapoints = list(test_loader)

        iterator = tqdm(range(len(datapoints)), desc="Sender")
        for xi in iterator:
            (x, _) = datapoints[xi]
            x = x.to(device).view(xdim)

            # calculate ELBO
            with torch.no_grad():
                model.compress(False)
                logrecon, logdec, logenc, _ = model.loss(x.view((-1,) + model.xs))
                elbo = -logrecon + torch.sum(-logdec + logenc)
                model.compress(True)

            # logging
            elbos[ei, xi] = elbo.item() / xdim

    print(f"E:{elbos.mean():.4f}±{elbos.std():.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)  # assign to gpu
    parser.add_argument('--nz', default=4, type=int)  # choose number of latent variables
    parser.add_argument('--quantbits', default=10, type=int)  # choose discretization precision
    parser.add_argument('--bitswap', default=1, type=int)  # choose whether to use Bit-Swap or not

    args = parser.parse_args()
    print(args)

    gpu = args.gpu
    nz = args.nz
    quantbits = args.quantbits
    bitswap = args.bitswap

    for nz in [nz]:
        for bits in [quantbits]:
            for bitswap in [bitswap]:
                eval_bpd(bits, nz, gpu)
