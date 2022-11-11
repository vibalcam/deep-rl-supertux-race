import torch
import torch.nn.functional as F


class KartCNN(torch.nn.Module):
    class BlockConv(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1, residual: bool = True):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
                # torch.nn.MaxPool2d(2, stride=2)
            )
            self.residual = residual
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.downsample is None else self.downsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    class BlockUpConv(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, residual: bool = True):
            super().__init__()
            # if kernel == 2:
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=2, stride=1, bias=False)
            # elif kernel == 3:
            #     # temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=stride,
            #     #                                 output_padding=1, bias=False)
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False)
            # elif kernel == 4:
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=4, padding=1, stride=1, bias=False)
            # else:
            #     raise Exception()

            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=stride, output_padding=1,
                                         bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.residual = residual
            self.upsample = None
            if stride != 1 or n_input != n_output:
                self.upsample = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=1, stride=stride, output_padding=1,
                                             bias=False),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.upsample is None else self.upsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    def __init__(self, dim_layers=[32, 64, 128], n_input=3, input_normalization: bool = True,
                 skip_connections: bool = False, residual: bool = True):
        super().__init__()

        n_output = 1

        self.skip_connections = skip_connections
        if input_normalization:
            self.norm = torch.nn.BatchNorm2d(n_input)
        else:
            self.norm = None

        self.min_size = 2 ** (len(dim_layers) + 1)

        c = dim_layers[0]
        self.net_conv = torch.nn.ModuleList([torch.nn.Sequential(
            # torch.nn.Conv2d(n_input, c, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.Conv2d(n_input, c, kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU()
        )])
        self.net_upconv = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(c * 2 if skip_connections else c, n_output, kernel_size=7,
                                     padding=3, stride=2, output_padding=1)
            # torch.nn.BatchNorm2d(5),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(5, 5, kernel_size=1)
        ])
        for k in range(len(dim_layers)):
            l = dim_layers[k]
            self.net_conv.append(self.BlockConv(c, l, stride=2, residual=residual))
            # Separate first upconv layer since it will never have an skip connection
            l = l * 2 if skip_connections and k != len(dim_layers) - 1 else l
            self.net_upconv.insert(0, self.BlockUpConv(l, c, stride=2, residual=residual))
            c = dim_layers[k]

    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        # Input Normalization
        if self.norm is not None:
            x = self.norm(x)

        h = x.size(2)
        w = x.size(3)

        if h < self.min_size or w < self.min_size:
            resize = torch.zeros([
                x.size(0),
                x.size(1),
                self.min_size if h < self.min_size else h,
                self.min_size if w < self.min_size else w
            ])
            # h_start = int((self.min_size - h) / 2 if h < self.min_size else 0)
            # w_start = int((self.min_size - w) / 2 if w < self.min_size else 0)
            # resize[:, :, h_start:h_start + h, w_start:w_start + w] = x
            resize[:, :, :h, :w] = x
            x = resize

        # Calculate
        partial_x = []
        for l in self.net_conv:
            x = l(x)
            partial_x.append(x)
        # Last one is not used for skip connections, skip after first upconv
        partial_x.pop(-1)
        skip = False
        for l in self.net_upconv:
            if skip and len(partial_x) > 0:
                x = torch.cat([x, partial_x.pop(-1)], 1)
                x = l(x)
            else:
                x = l(x)
                skip = self.skip_connections

        return spatial_argmax(x[:, 0, :h, :w])

