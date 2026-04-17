import torch

from pf_dea_net import PFDEANet


def main() -> None:
    model = PFDEANet(base_dim=32)
    x = torch.rand(2, 3, 256, 256)
    y = model(x)
    print("Output keys:", list(y.keys()))
    for k, v in y.items():
        print(k, tuple(v.shape))


if __name__ == "__main__":
    main()
