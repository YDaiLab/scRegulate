import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE model
class scRNA_VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for scRNA-seq data.

    Args:
        input_dim (int): Number of input features (genes).
        encode_dims (list): List of hidden layer sizes for the encoder.
        decode_dims (list): List of hidden layer sizes for the decoder.
        z_dim (int): Size of the latent space.
        tf_dim (int): Size of the transcription factor space.
        ulm_init (torch.Tensor, optional): Pre-initialized TF activity matrix.
    """

    def __init__(self, input_dim, encode_dims, decode_dims, z_dim, tf_dim, ulm_init=None):
        super(scRNA_VAE, self).__init__()
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        previous_dim = input_dim
        for h_dim in encode_dims:
            self.encoder_layers.append(nn.Linear(previous_dim, h_dim))
            previous_dim = h_dim
        self.fc_mu = nn.Linear(previous_dim, z_dim)
        self.fc_logvar = nn.Linear(previous_dim, z_dim)
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        previous_dim = z_dim
        for h_dim in decode_dims:
            self.decoder_layers.append(nn.Linear(previous_dim, h_dim))
            previous_dim = h_dim
        self.fc_output = nn.Linear(previous_dim, tf_dim)
        self.tf_mapping = nn.Linear(tf_dim, input_dim)
        self.ulm_init = ulm_init


    def encode(self, x):
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for layer in self.decoder_layers:
            z = F.relu(layer(z))
        return F.relu(self.fc_output(z))  # Was torch.exp()

    def forward(self, x, tf_activity_init=None, alpha=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
    
        if tf_activity_init is not None:
            # Use a weighted combination of pre-initialized and learned TF activities
            tf_activity = (1 - alpha) * tf_activity_init.detach() + alpha * self.decode(z)
        else:
            # If no pre-initialized TF activities, just use the decoded values
            tf_activity = self.decode(z)
            
        recon_x = self.tf_mapping(tf_activity)
        return recon_x, mu, logvar


    def encodeBatch(self, dataloader, device='cuda', out='z'):
        output = []
        for batch in dataloader:
            batch = batch.to(device)
            mu, logvar = self.encode(batch)
            z = self.reparameterize(mu, logvar)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                recon_x = self.tf_mapping(self.decode(z))
                output.append(recon_x.detach().cpu())
            elif out == 'tf':
                tf_activity = self.decode(z)
                output.append(tf_activity.detach().cpu())
        
        output = torch.cat(output).numpy()
        return output

