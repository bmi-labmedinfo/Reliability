import torch
from sklearn.metrics import mean_squared_error


class CosineActivation(torch.nn.Module):
    """
    A custom activation function that applies the cosine function.

    The CosineActivation class is a PyTorch module that applies the cosine activation function
    to the input tensor.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(x): Applies the cosine activation to the input tensor.
  """

    def __init__(self):
        """
        Initializes an instance of the CosineActivation class.

        Args:
            None
    """
        super().__init__()

    def forward(self, x):
        """
        Applies the cosine activation to the input tensor.

        The forward method takes an input tensor and applies the cosine activation function
        element-wise, subtracting the input value from its cosine.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the cosine activation applied.
    """
        return torch.cos(x) - x


class AE(torch.nn.Module):
    """
    Autoencoder model implemented as a PyTorch module.

    The AE class represents an autoencoder model with specified sizes of the layers.
    It consists of an encoder and a decoder, both utilizing the CosineActivation as
    the activation function.

    Args:
        layer_sizes (list): a list containing the sizes of the layers of the encoder (decoder built with symmetry).

    Attributes:
        encoder (torch.nn.Sequential): The encoder module.
        decoder (torch.nn.Sequential): The decoder module.

    Methods:
        forward(x): Performs the forward pass of the autoencoder model.
  """

    def __init__(self, layer_sizes):
        """
        Initializes an instance of the AE class.

        Args:
            layer_sizes (list): A list of integers containing the sizes of the layers.
    """
        super().__init__()
        self.encoder = self.build_encoder(layer_sizes)
        self.decoder = self.build_decoder(layer_sizes)

    def build_encoder(self, layer_sizes):
        """
        Builds the encoder part of an autoencoder model based on the specified layer sizes.

        Parameters:
            layer_sizes (list): A list of integers representing the number of nodes in each layer of the encoder.

        Returns:
            encoder (torch.nn.Sequential): The encoder module of the autoencoder model.
    """
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(CosineActivation())
        return torch.nn.Sequential(*encoder_layers)

    def build_decoder(self, layer_sizes):
        """
        Builds the decoder part of an autoencoder model based on the specified layer sizes.

        Parameters:
            layer_sizes (list): A list of integers representing the number of nodes in each layer of the decoder.

        Returns:
            decoder (torch.nn.Sequential): The decoder module of the autoencoder model.
    """
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            decoder_layers.append(CosineActivation())
        return torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Performs the forward pass of the autoencoder model.

        The forward method takes an input tensor and passes it through the encoder,
        obtaining the encoded representation. The encoded representation is then passed
        through the decoder to reconstruct the original input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The reconstructed tensor.
    """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ReliabilityDetector:
    """
    Reliability Detector for assessing the reliability of data points.

    The ReliabilityPackage class computes the reliability of data points based on
    a specified autoencoder (ae), a proxy model (clf), and an MSE threshold (mse_thresh).

    Args:
        ae (AE): The autoencoder model.
        proxy_model: The proxy model used for the local fit reliability computation.
        mse_thresh (float): The MSE threshold used for the density reliability computation.

    Attributes:
        ae (AE): The autoencoder model.
        clf: The proxy model used for the local fit reliability computation.
        mse_thresh (float): The MSE threshold for the density reliability computation.

    Methods:
        compute_density_reliability(x): Computes the density reliability of a data point.
        compute_localfit_reliability(x): Computes the local fit reliability of a data point.
        compute_total_reliability(x): Computes the combined reliability of a data point.
  """

    def __init__(self, ae, proxy_model, mse_thresh):
        """
        Initializes an instance of the ReliabilityPackage class.

        Args:
            ae (AE): The autoencoder model.
            proxy_model: The proxy model used for the local fit reliability computation.
            mse_thresh (float): The MSE threshold used for the density reliability computation.
    """
        self.ae = ae
        self.clf = proxy_model
        self.mse_thresh = mse_thresh

    def compute_density_reliability(self, x):
        """
        Computes the density reliability of a data point.

        The density reliability is determined by computing the mean squared error (MSE)
        between the input data point and its reconstructed representation obtained from
        the autoencoder. If the MSE is less than (or equal to) the specified MSE threshold,
        the data point is considered reliable (returns 1), otherwise unreliable (returns 0).

        Args:
            x (numpy.ndarray or torch.Tensor): The input data point.

        Returns:
            int: The density reliability value (1 for reliable, 0 for unreliable).
    """
        mse = mean_squared_error(x, self.ae((torch.tensor(x)).float()).detach().numpy())
        return 1 if mse <= self.mse_thresh else 0

    def compute_localfit_reliability(self, x):
        """
        Computes the local fit reliability of a data point.

        The local fit reliability is determined by using the proxy model to predict the local fit
        reliability of the input data point. The input data point is reshaped to match the
        expected input format of the proxy model. The predicted reliability value is returned.

        Args:
            x (numpy.ndarray or torch.Tensor): The input data point.

        Returns:
            int: The local fit reliability class predicted by the proxy model (1 for reliable, 0 for unreliable).
    """
        return self.clf.predict(x.reshape(1, -1))[0]

    def compute_total_reliability(self, x):
        """
        Computes the combined reliability of a data point.

        The combined reliability is determined by combining the density reliability and the
        local fit reliability. If both reliabilities are positive (1), the data point is
        considered reliable (returns True), otherwise unreliable (returns False).

        Args:
            x (numpy.ndarray or torch.Tensor): The input data point.

        Returns:
            bool: The combined reliability value (True for reliable, False for unreliable).
    """
        density_rel = self.compute_density_reliability(x)
        localfit_rel = self.compute_localfit_reliability(x)
        return density_rel and localfit_rel


class DensityPrincipleDetector:
    """
    Density principle Detector for assessing the density reliability of data points.

    The DensityPrincipleDetector class computes the density reliability of data points based on
    a specified autoencoder (autoencoder) and a threshold (threshold).

    Args:
        autoencoder: The autoencoder model.
        threshold (float): The threshold for determining the density reliability.

    Attributes:
        ae: The autoencoder model.
        thresh (float): The threshold for determining the density reliability.

    Methods:
        compute_reliability(x): Computes the density reliability of a data point.
  """

    def __init__(self, autoencoder, threshold):
        """
        Initializes an instance of the DensityPrincipleDetector class.

        Args:
            autoencoder: The autoencoder model.
            threshold (float): The threshold for determining the density reliability.
    """
        self.ae = autoencoder
        self.thresh = threshold

    def compute_reliability(self, x):
        """
        Computes the density reliability of a data point.

        The density reliability is determined by computing the mean squared error (MSE)
        between the input data point and its reconstructed representation obtained from
        the autoencoder. If the MSE is less than or equal to the specified threshold,
        the data point is considered reliable (returns 1), otherwise unreliable (returns 0).

        Args:
            x (numpy.ndarray or torch.Tensor): The input data point.

        Returns:
            int: The density reliability value (1 for reliable, 0 for unreliable).
    """
        mse = mean_squared_error(x, self.ae((torch.tensor(x)).float()).detach().numpy())
        return 1 if mse <= self.thresh else 0
