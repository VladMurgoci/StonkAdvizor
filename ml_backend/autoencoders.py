import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

METRICS_COUNT = 51

class Autoencoder(torch.nn.Module): 
    def __init__(self, input_size: int, first_hidden_layer_size: int, latent_repr_size: int, reduction: float) -> None:
        super().__init__()

        self.cluster_info = {}
        encoder_modules = [] 
        depth = -1
        while True:
            depth += 1
            i_size = int(first_hidden_layer_size // (reduction ** depth))
            o_size = int(first_hidden_layer_size // (reduction ** (depth + 1)))
            print(f"i_size: {i_size}, o_size: {o_size}")
            if o_size <= latent_repr_size:
                break
            encoder_modules.append(torch.nn.Linear(i_size, o_size))
            encoder_modules.append(torch.nn.ReLU())
        encoder_modules.append(torch.nn.Linear(int(first_hidden_layer_size // (reduction ** depth)), latent_repr_size))
        encoder_modules.insert(0, torch.nn.ReLU())
        encoder_modules.insert(0, torch.nn.Linear(input_size, first_hidden_layer_size))
        self.encoder = torch.nn.Sequential(*encoder_modules)

        encoder_shapes = [layer.weight.shape for idx, layer in enumerate(self.encoder) if idx % 2 == 0]
        print(encoder_shapes)

        decoder_modules = [] 
        for i in range(0, len(self.encoder), 2):
            reversed_shape = self.encoder[len(self.encoder) - i - 1].weight.shape
            decoder_modules.append(torch.nn.Linear(reversed_shape[0], reversed_shape[1]))
            if i == len(self.encoder) - 2:
                break
            decoder_modules.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*decoder_modules)
        decoder_shapes = [layer.weight.shape for idx, layer in enumerate(self.decoder) if idx % 2 == 0]
        print(decoder_shapes)


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
    def train(self, data, loss_f, optim, n_epochs=100, batch_size=32):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_data, _ in data_loader:
                optim.zero_grad()
                reconstructed = self.forward(batch_data)
                loss = loss_f(reconstructed, batch_data)
                loss.backward()
                optim.step()
                epoch_loss += loss.item() * len(batch_data)
            epoch_loss /= len(data.tensors[0])
            if epoch == n_epochs - 1:
                self._model_loss = epoch_loss
            print(f"Epoch {epoch} loss: {epoch_loss}")
        print("Training finished")

    def encoder_pass(self, data):
        return self.encoder(data)

    def cluster_latent_repr(self, data: torch.Tensor, n_clusters: int) -> None:
        encoded_data = self.encoder_pass(data).detach().numpy()
        clustered_data = KMeans(n_clusters=n_clusters, random_state=0).fit(encoded_data)
        for cluster_idx in range(n_clusters):
            indices = np.where(clustered_data.labels_ == cluster_idx)[0]
            distances = self.euclidean_distance(clustered_data.cluster_centers_[cluster_idx], encoded_data[indices])
            self.cluster_info[cluster_idx] = (clustered_data.cluster_centers_[cluster_idx], distances.max(), distances.min(), distances.mean())
        
    def classify_sample(self, sample: torch.Tensor) -> int:
        encoded_sample = self.encoder_pass(sample).detach().numpy()
        closest_cluster, distance_to_closest_cluster = None, None
        for cluster_idx, (c_center, dist_to_c_center, _, _) in self.cluster_info.items():
            distance = self.euclidean_distance(encoded_sample, c_center)
            if dist_to_c_center > distance and (distance_to_closest_cluster is None or distance_to_closest_cluster > distance):
                closest_cluster = cluster_idx
                distance_to_closest_cluster = distance

        return closest_cluster, distance_to_closest_cluster
        
        
    def euclidean_distance(self, x, y):
        return (x - y) ** 2 

    def test(self, data):
        with torch.no_grad():
            reconstructed = self.forward(data)
            return reconstructed

    @property
    def model_loss(self):
        return self._model_loss

    @model_loss.setter
    def model_loss(self, value):
        self._model_loss = value