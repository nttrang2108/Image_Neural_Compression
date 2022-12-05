import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_digits

from tqdm import tqdm
import numpy as np
import wandb
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

EPS = 1.0e-7


class Digit_Dataset(Dataset):
    def __init__(self, mode="train", transform=None):
        digits = load_digits()
        if mode == "train":
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == "val":
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class Encoder(nn.Module):
    def __init__(self, D, M, C):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(D, M * 2),
            nn.BatchNorm1d(M * 2),
            nn.ReLU(),
            nn.Linear(M * 2, M),
            nn.BatchNorm1d(M),
            nn.ReLU(),
            nn.Linear(M, M // 2),
            nn.BatchNorm1d(M // 2),
            nn.ReLU(),
            nn.Linear(M // 2, C),
        )

    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(self, D, M, C):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(C, M // 2),
            nn.BatchNorm1d(M // 2),
            nn.ReLU(),
            nn.Linear(M // 2, M),
            nn.BatchNorm1d(M),
            nn.ReLU(),
            nn.Linear(M, M * 2),
            nn.BatchNorm1d(M * 2),
            nn.ReLU(),
            nn.Linear(M * 2, D),
        )

    def forward(self, x):
        return self.decode(x)


class Quantizer(nn.Module):
    def __init__(self, input_dim, codebook_dim, temp=1.0e7):
        super(Quantizer, self).__init__()
        self.temp = temp
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.codebook = nn.Parameter(
            torch.FloatTensor(
                1,
                self.codebook_dim,
            ).uniform_(-1 / self.codebook_dim, 1 / self.codebook_dim)
        )

    def indices2codebook(self, indices_onehot):
        return torch.matmul(indices_onehot, self.codebook.t()).squeeze()

    def indices_to_onehot(self, inputs_shape, indices):
        indices_hard = torch.zeros(inputs_shape[0], inputs_shape[1], self.codebook_dim)
        indices_hard.scatter_(2, indices, 1)

    def forward(self, inputs):
        inputs_shape = inputs.shape
        inputs_repeat = inputs.unsqueeze(2).repeat(1, 1, self.codebook_dim)
        distances = torch.exp(
            -torch.sqrt(torch.pow(inputs_repeat - self.codebook.unsqueeze(1), 2))
        )
        indices = torch.argmax(distances, dim=2).unsqueeze(2)
        indices_hard = self.indices_to_onehot(
            inputs_shape=inputs_shape, indices=indices
        )
        indices_soft = torch.softmax(self.temp * distances, -1)
        quantized = self.indices2codebook(indices_onehot=indices_soft)
        return (indices_soft, indices_hard, quantized)


class Uniform_Entropy_Coding(nn.Module):
    def __init__(self, code_dim, codebook_dim):
        super(Uniform_Entropy_Coding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim
        self.probs = torch.softmax(torch.ones(1, self.code_dim, self.codebook_dim), -1)

    def sample(self, quantizer=None, B=10):
        code = torch.zeros(B, self.code_dim, self.codebook_dim)
        for b in range(B):
            indx = torch.multinomial(
                torch.softmax(self.probs, -1).squeeze(0), 1
            ).squeeze()
            for i in range(self.code_dim):
                code[b, i, indx[i]] = 1

        code = quantizer.indices2codebook(code)
        return code

    def forward(self, z, x=None):
        p = torch.clamp(self.probs, EPS, 1.0 - EPS)
        return -torch.sum(z * torch.log(p), 2)


class Independent_Entropy_Coding(nn.Module):
    def __init__(self, code_dim, codebook_dim):
        super(Independent_Entropy_Coding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim

        self.probs = nn.Parameter(torch.ones(1, self.code_dim, self.codebook_dim))

    def sample(self, quantizer=None, B=10):
        code = torch.zeros(B, self.code_dim, self.codebook_dim)
        for b in range(B):
            indx = torch.multinomial(
                torch.softmax(self.probs, -1).squeeze(0), 1
            ).squeeze()
            for i in range(self.code_dim):
                code[b, i, indx[i]] = 1

        code = quantizer.indices2codebook(code)
        return code

    def forward(self, z, x=None):
        p = torch.clamp(torch.softmax(self.probs, -1), EPS, 1.0 - EPS)
        return -torch.sum(z * torch.log(p), 2)


class Casual_Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        stride=1,
        A=False,
        **kwargs
    ):
        super(Casual_Conv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A
        self.padding = (kernel_size - 1) * dilation + A * 1
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        x = self.conv1d(x)
        if self.A:
            return x[:, :, :-1]
        else:
            return x


class ARM_Entropy_Coding(nn.Module):
    def __init__(self, code_dim, codebook_dim, E=8, M_kernels=32, kernel=4):
        super(ARM_Entropy_Coding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim
        self.arm_net = nn.Sequential(
            Casual_Conv1d(
                in_channels=1,
                out_channels=M_kernels,
                dilation=1,
                kernel_size=kernel,
                A=True,
                bias=True,
            ),
            nn.LeakyReLU(),
            Casual_Conv1d(
                in_channels=M_kernels,
                out_channels=M_kernels,
                dilation=1,
                kernel_size=kernel,
                A=False,
                bias=True,
            ),
            nn.LeakyReLU(),
            Casual_Conv1d(
                in_channels=M_kernels,
                out_channels=E,
                dilation=1,
                kernel_size=kernel,
                A=False,
                bias=True,
            ),
        )

    def f(self, x):
        h = self.arm_net(x.unsqueeze(1))
        # print("h_size", h.size())
        h = h.permute(0, 2, 1)
        # print("h_size", h.size())
        p = torch.softmax(h, 2)
        # print("p_size", p.size())

        return p

    def sample(self, quantizer=None, B=10):
        x_new = torch.zeros((B, self.code_dim))

        for d in range(self.code_dim):
            p = self.f(x_new)
            indx_d = torch.multinomial(p[:, d, :], num_samples=1)
            codebook_value = quantizer.codebook[0, indx_d].squeeze()
            x_new[:, d] = codebook_value

        return x_new

    def forward(self, z, x):
        p = self.f(x)
        return -torch.sum(z * torch.log(p), 2)


class Neural_Compressor(nn.Module):
    def __init__(
        self, encoder, decoder, entropy_coding, quantizer, beta=1, detaching=False
    ):
        super(Neural_Compressor, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.entropy_coding = entropy_coding
        self.quantizer = quantizer
        self.beta = beta
        self.detaching = detaching

    def forward(self, x, reduction="avg"):
        z = self.encoder(x)
        quantizer_out = self.quantizer(z)
        x_rec = self.decoder(quantizer_out[2])

        Distortion = torch.mean(torch.pow(x - x_rec, 2), (1))

        Rate = torch.mean(self.entropy_coding(quantizer_out[0], quantizer_out[2]), 1)
        objective = Distortion + self.beta * Rate

        if reduction == "sum":
            return objective.sum(), Distortion.sum(), Rate.sum()
        else:
            return objective.mean(), Distortion.mean(), Rate.mean()


class Training_Model:
    def __init__(self, model, optimizer, dict_dataloader):
        # self.model = model.to(device)
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dict_dataloader["train_loader"]
        self.val_loader = dict_dataloader["val_loader"]
        self.best_val_loss = 1000

    def train(self, epochs):
        print("============TRAINING START {}============".format(epochs))
        self.model.train()

        total_loss = 0
        total_distortion = 0
        total_rate = 0
        pbar = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(pbar):
            if hasattr(self.model, "dequantization"):
                if self.model.dequantization:
                    batch = batch + torch.rand(batch.shape)

            self.optimizer.zero_grad()
            # batch = batch.to(device)
            loss, distortion, rate = self.model(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_distortion += distortion.item()
            total_rate += rate.item()
            pbar.set_description(
                "Train_Loss: {:.4f} Train_Distortion: {:.4f} Train_Rate: {:.4f}".format(
                    loss, distortion, rate
                )
            )

        total_loss = round(total_loss / len(self.train_loader), 4)
        total_distortion = round(total_distortion / len(self.train_loader), 4)
        total_rate = round(total_rate / len(self.train_loader), 4)

        wandb.log(
            {
                "train_loss": total_loss,
                "train_distortion": total_distortion,
                "train_rate": total_rate,
            }
        )

    def eval(self, epochs, max_patience):
        print("============EVALUATION START {}============".format(epochs))
        self.model.eval()
        total_loss = 0
        total_distortion = 0
        total_rate = 0
        patience = 0
        pbar = tqdm(self.val_loader)
        for batch_idx, batch in enumerate(pbar):

            # to device
            # batch = batch.to(device)
            loss, distortion, rate = self.model(batch, reduction="sum")
            total_loss += loss.item()
            total_distortion += distortion.item()
            total_rate += rate.item()
            pbar.set_description(
                "Val_Loss: {:.4f} Val_Distortion: {:.4f} Val_Rate: {:.4f}".format(
                    loss, distortion, rate
                )
            )

        total_loss = round(total_loss / len(self.val_loader.dataset), 4)
        total_distortion = round(total_distortion / len(self.val_loader.dataset), 4)
        total_rate = round(total_rate / len(self.val_loader.dataset), 4)

        if total_loss < self.best_val_loss:
            self.best_val_loss = total_loss
            # save model
            torch.save(
                self.model,
                "../weights/img_8/img_28_indp/best_model_{}_{}.pt".format(
                    epochs, total_loss
                ),
            )
            print("Save best model epoch {} total_loss{}".format(epochs, total_loss))
        else:
            patience += 1

        if patience == max_patience:
            print(">>>Early stopping - Max patience reached<<<")

        wandb.log(
            {
                "loss_val": total_loss,
                "distortion_val": total_distortion,
                "rate_val": total_rate,
            }
        )


def config_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_coding_type", type=str, default="arm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--C", type=int, default=16)
    parser.add_argument("--E", type=int, default=8)
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--M_kernels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dequantization", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max_patience", type=int, default=50)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_input()
    config_seed(args.seed)

    train_data = Digit_Dataset(mode="train")
    val_data = Digit_Dataset(mode="val")
    test_data = Digit_Dataset(mode="test")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    dict_dataloader = {"train_loader": train_loader, "val_loader": val_loader}

    if args.entropy_coding_type == "uniform":
        beta = 0.0
    else:
        beta = 1.0

    encoder = Encoder(D=args.D, M=args.M, C=args.C)

    decoder = Decoder(D=args.D, M=args.M, C=args.C)

    quantizer = Quantizer(input_dim=args.C, codebook_dim=args.E)

    if args.entropy_coding_type == "uniform":
        entropy_coding = Uniform_Entropy_Coding(code_dim=args.C, codebook_dim=args.E)
    elif args.entropy_coding_type == "indp":
        entropy_coding = Independent_Entropy_Coding(
            code_dim=args.C, codebook_dim=args.E
        )
    else:
        kernel = 4
        entropy_coding = ARM_Entropy_Coding(
            code_dim=args.C,
            codebook_dim=args.E,
            E=args.E,
            M_kernels=args.M_kernels,
            kernel=kernel,
        )

    model = Neural_Compressor(
        encoder=encoder,
        decoder=decoder,
        entropy_coding=entropy_coding,
        quantizer=quantizer,
        beta=beta,
        detaching=False,
    )

    optimizer = torch.optim.Adamax(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    wandb.init(
        project="Image_Compressor",
        group="Neural_Compressor",
        name="neural_compressor_{}_{}_{}".format(
            args.entropy_coding_type, args.epochs, args.lr
        ),
    )

    wandb.config.update(args)

    training_model = Training_Model(
        model=model, optimizer=optimizer, dict_dataloader=dict_dataloader
    )

    for epoch in range(args.epochs):
        training_model.train(epoch)
        training_model.eval(epoch, args.max_patience)
