import numpy as np
import torch
from torch.utils.data import Dataset
import os

class SingleTransformerDataset(Dataset):
    def __init__(self, data_dir):
        self.adj = np.load(os.path.join(data_dir, "adjacency.npy"), mmap_mode='r')
        self.curves = np.load(os.path.join(data_dir, "curves.npy"), mmap_mode='r')
        self.dec_in = np.load(os.path.join(data_dir, "decoder_input.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')
        self.masks = np.load(os.path.join(data_dir, "masks.npy"), mmap_mode='r')

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        return {
            "adjacency": torch.tensor(self.adj[idx], dtype=torch.float32),
            "curve_numerical": torch.tensor(self.curves[idx], dtype=torch.float32),
            "decoder_input": torch.tensor(self.dec_in[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "decoder_mask": torch.tensor(self.masks[idx], dtype=torch.bool),
        }

class SingleImageTransformerDataset(Dataset):
    def __init__(self, data_dir):
        # Load all data into RAM (no memory mapping)
        self.adj = np.load(os.path.join(data_dir, "adjacency.npy"))
        self.curves = np.load(os.path.join(data_dir, "curve_images.npy"))
        self.dec_in = np.load(os.path.join(data_dir, "decoder_input.npy"))
        self.labels = np.load(os.path.join(data_dir, "labels.npy"))
        self.masks = np.load(os.path.join(data_dir, "masks.npy"))
        
        # Precompute length
        self.length = len(self.curves)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # No .copy() needed since data is already in RAM and writable
        # No np.array() needed either
        curve_np = self.curves[idx]  # Shape: [64, 64, 1]
        
        # Convert curve from [64, 64, 1] to [1, 64, 64]
        curve_tensor = torch.from_numpy(curve_np).float()
        curve_tensor = curve_tensor.permute(2, 0, 1).contiguous()  # [1, 64, 64]
        
        return {
            "adjacency": torch.from_numpy(self.adj[idx]).float(),
            "curve_numerical": curve_tensor,
            "decoder_input": torch.from_numpy(self.dec_in[idx]).float(),
            "label": torch.from_numpy(self.labels[idx]).float(),
            "decoder_mask": torch.from_numpy(self.masks[idx]).bool(),
        }
    
# class SingleTransformerDataset(Dataset):
#     def __init__(self, node_features_path, edge_index_path, curves_path, max_nodes=20, shuffle=True):
#         self.node_features = np.load(node_features_path, allow_pickle=True)
#         self.edge_index = np.load(edge_index_path, allow_pickle=True)
#         self.curves = np.load(curves_path, mmap_mode='r')
#         self.max_nodes = max_nodes
#         self.max_sequence_length = 10  # for decoder sequence
#         if shuffle:
#             self._shuffle_data()

#     def _shuffle_data(self):
#         indices = np.random.permutation(len(self.node_features))
#         self.node_features = self.node_features[indices]
#         self.edge_index = self.edge_index[indices]
#         self.curves = self.curves[indices]

#     def __len__(self):
#         return len(self.node_features)

#     def __getitem__(self, idx):
#         raw_node = self.node_features[idx]             # shape: [n, 3]
#         edge_idx = self.edge_index[idx]                # shape: [2, num_edges]
#         curve = torch.tensor(self.curves[idx], dtype=torch.float32)  # [200, 2]

#         node_feats = raw_node[2:, :2]                  # shape: [n, 2]
#         node_attr = raw_node[:, 2]                     # shape: [n]
#         n = len(node_feats)

#         # Build adjacency
#         adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
#         for i, j in zip(edge_idx[0], edge_idx[1]):
#             adj[i, j] = 1.0
#             adj[j, i] = 1.0
#         for i in range(min(n, self.max_nodes)):
#             adj[i, i] = node_attr[i]
#         adjacency_tensor = torch.tensor(adj, dtype=torch.float32).unsqueeze(0)

#         # Create single decoder input/label
#         pos = torch.tensor(node_feats, dtype=torch.float32)
#         decoder_input, label = self._create_decoder_data(pos)
#         mask = self._create_combined_mask(decoder_input)

#         return {
#             "adjacency": adjacency_tensor,
#             "curve_numerical": curve,
#             "decoder_input": decoder_input,
#             "label": label,
#             "decoder_mask": mask,
#         }

#     def _create_decoder_data(self, pos):
#         # Insert <s> and pad up to max_sequence_length
#         num_nodes = pos.size(0)
#         pad_len = self.max_sequence_length - num_nodes - 1
#         decoder_input = torch.cat([
#             torch.ones(1, pos.size(1)) * -2.0,                  # <s>
#             pos,
#             torch.full((pad_len, pos.size(1)), -1.0)
#         ], dim=0)
#         label = torch.cat([
#             pos,
#             torch.ones(1, pos.size(1)),                         # </s>
#             torch.full((pad_len, pos.size(1)), -1.0)
#         ], dim=0)
#         return decoder_input, label

#     def _create_combined_mask(self, mech):
#         size = mech.size(0)
#         causal_mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
#         pad_mask = ~(mech == -1.0).all(dim=1)
#         pad_mask = pad_mask.unsqueeze(0).expand(size, -1)
#         combined_mask = causal_mask | ~pad_mask | ~pad_mask.T
#         return ~combined_mask
