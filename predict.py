import csv
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn

# get data
from dataset import HW3Dataset  # TODO: is that true?
dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]
# enrich dataset

# Convert the Data object to a NetworkX graph
graph = to_networkx(data)
# Compute degree for each node
degrees = torch.tensor([graph.degree[node] for node in range(data.num_nodes)])
# Compute other measures and attributes
clustering = nx.clustering(graph)
eigenvector_centrality = nx.eigenvector_centrality(graph)
pagerank = nx.pagerank(graph)
degree_centrality = nx.degree_centrality(graph)
katz_centrality = nx.katz_centrality(graph)

in_degrees = torch.zeros(data.num_nodes)
out_degrees = torch.zeros(data.num_nodes)
# Iterate over the edges and update the degrees
for src, tgt in data.edge_index.t().tolist():
    out_degrees[src] += 1
    in_degrees[tgt] += 1

year = data.node_year


def normalize(values):
    # Min-max normalization
    min_value = torch.min(values)
    max_value = torch.max(values)
    return (values - min_value) / (max_value - min_value)


year = normalize(year)
degrees = normalize(degrees)
in_degrees = normalize(in_degrees)
out_degrees = normalize(out_degrees)
clustering = normalize(torch.tensor(list(clustering.values())))
eigenvector_centrality = normalize(torch.tensor(list(eigenvector_centrality.values())))
pagerank = normalize(torch.tensor(list(pagerank.values())))
degree_centrality = normalize(torch.tensor(list(degree_centrality.values())))
katz_centrality = normalize(torch.tensor(list(katz_centrality.values())))

data.x = torch.cat([data.x
                    ,year.view(-1, 1)
                    ,degrees.view(-1, 1)
                    ,in_degrees.view(-1, 1)
                    ,out_degrees.view(-1, 1)
                    ,clustering.view(-1, 1)
                    ,eigenvector_centrality.view(-1, 1)
                    ,pagerank.view(-1, 1)
                    ,degree_centrality.view(-1, 1)
                    ,katz_centrality.view(-1, 1)
                    ], dim=1)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(137, hidden_channels) # Noam: it was dataset.num_features and I changed to 137.
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.batch_norm = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x_in = x
        x = self.conv(x, edge_index)
        x = x + x_in
        x = self.batch_norm(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x

# model
model = GCN(hidden_channels=512)
model.load_state_dict(torch.load('model_to_submit.pt'))
model.eval()


out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)  # Use the class with highest probability.

# Create a CSV file and write the header
with open('predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['idx', 'prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Write the data rows
    for idx, prediction in enumerate(pred, start=0):
        writer.writerow({'idx': idx, 'prediction': prediction.item()})

print("DONE!")