import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
import numpy as np

# embedding model will be mpnet-base-v2
class KnowledgeGraphEmbeddings:
    def __init__(self, graph=None, embedding_dim=100, text_model="all-mpnet-base-v2"):
        """
        Initialize the KnowledgeGraphEmbeddings class.

        Args:
            graph (networkx.DiGraph): The Knowledge Graph as a NetworkX DiGraph.
            embedding_dim (int): Dimension of the embeddings for structural representations.
            text_model (str): Pretrained text embedding model (default: SentenceTransformer).
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.text_model = SentenceTransformer(text_model)  # Load pre-trained text embedding model
        self.node_embeddings = {}
        self.edge_embeddings = {}
        self.combined_embeddings = {}
        self.structural_embeddings = {}
        self.textual_embeddings = {}

    def generate_structural_embeddings(self):
        """
        Generate structural embeddings using TransE.
        """
        # Convert the graph into a list of triples
        triples = [(src, rel, tgt) for src, tgt, rel in self.graph.edges(data="relation")]

        # Create a PyKEEN TriplesFactory
        triples_factory = TriplesFactory.from_labeled_triples(
            [(str(h), str(r), str(t)) for h, r, t in triples]
        )

        # Train TransE model
        model = TransE(triples_factory=triples_factory, embedding_dim=self.embedding_dim)
        model.train(num_epochs=50)

        # Store entity (node) embeddings
        for entity in self.graph.nodes:
            self.node_embeddings[entity] = model.entity_representations[0](
                torch.tensor([triples_factory.entity_to_id[entity]])
            ).detach().numpy().squeeze()

        # Store relation (edge) embeddings
        for relation in self.graph.edges:
            self.edge_embeddings[relation] = model.relation_representations[0](
                torch.tensor([triples_factory.relation_to_id[relation]])
            ).detach().numpy().squeeze()

        # Combine entity and relation embeddings
        for node, edge in self.graph.edges:
            self.combined_embeddings[(node, edge)] = (
                self.node_embeddings[node] + self.edge_embeddings[edge]
            )

        # Store combined embeddings
        for node, edge, combined in self.graph.edges:
            self.structural_embeddings[(node, edge)] = combined

        print("Structural embeddings generated.")
        return self.structural_embeddings

    def generate_textual_embeddings(self, text_field="chunk_text"):
        """
        Generate textual embeddings for each node based on the provided text field.

        Args:
            text_field (str): The metadata field in the graph containing textual content.
        """
        for node, data in self.graph.nodes(data=True):
            text = data.get(text_field, "Unknown")
            self.textual_embeddings[node] = self.text_model.encode(text)
        
        print("Textual embeddings generated.")
        return self.textual_embeddings

    def combine_embeddings(self, strategy="concat"):
        """
        Combine structural and textual embeddings.

        Args:
            strategy (str): Strategy for combining embeddings ('concat' or 'average').

        Returns:
            dict: Combined embeddings for each node.
        """
        combined_embeddings = {}
        for node in self.graph.nodes:
            struct_emb = self.structural_embeddings.get(node, np.zeros(self.embedding_dim))
            text_emb = self.textual_embeddings.get(node, np.zeros(self.text_model.get_sentence_embedding_dimension()))

            if strategy == "concat":
                combined_embeddings[node] = np.concatenate([struct_emb, text_emb])
            elif strategy == "average":
                combined_embeddings[node] = (struct_emb + text_emb) / 2

        self.combined_embeddings = combined_embeddings
        print(f"Combined embeddings generated using {strategy} strategy.")
        return self.combined_embeddings

    def to_torch_geometric_data(self):
        """
        Convert the graph to a PyTorch Geometric Data object for further processing.

        Returns:
            Data: A PyTorch Geometric Data object.
        """
        edge_index = torch.tensor(
            [(src, tgt) for src, tgt, _ in self.graph.edges(data=True)], dtype=torch.long
        ).t().contiguous()

        x = torch.tensor(
            [self.combined_embeddings[node] for node in self.graph.nodes],
            dtype=torch.float
        )

        return Data(x=x, edge_index=edge_index)

    def save_embeddings(self, output_path, embedding_type="combined"):
        """
        Save embeddings to a file.

        Args:
            output_path (str): Path to save the embeddings.
            embedding_type (str): Type of embeddings to save ('structural', 'textual', or 'combined').
        """
        if embedding_type == "structural":
            embeddings = self.structural_embeddings
        elif embedding_type == "textual":
            embeddings = self.textual_embeddings
        elif embedding_type == "combined":
            embeddings = self.combined_embeddings
        else:
            raise ValueError("Invalid embedding type. Choose from 'structural', 'textual', or 'combined'.")

        with open(output_path, "w") as f:
            for node, emb in embeddings.items():
                emb_str = " ".join(map(str, emb))
                f.write(f"{node} {emb_str}\n")

        print(f"{embedding_type.capitalize()} embeddings saved to {output_path}.")


if __name__ == "__main__":
    import networkx as nx
    from knowledge_graph.kg_dataset import KnowledgeGraphDataset  

    # Load the graph
    kg_dataset = KnowledgeGraphDataset(json_path="/Users/sidgraph/Desktop/KG-Hallucination-RAG/faith_kg_llm_project/data/train.json")
    kg_dataset.construct_graph()
    G = kg_dataset.to_networkx()

    # Initialize the KnowledgeGraphEmbeddings
    kg_embeddings = KnowledgeGraphEmbeddings(graph=G, embedding_dim=100)

    # Generate structural and textual embeddings
    structural_emb = kg_embeddings.generate_structural_embeddings()
    textual_emb = kg_embeddings.generate_textual_embeddings()

    # Combine the embeddings
    combined_emb = kg_embeddings.combine_embeddings(strategy="concat")

    # Convert to PyTorch Geometric Data
    data = kg_embeddings.to_torch_geometric_data()

    # Save embeddings
    kg_embeddings.save_embeddings("combined_embeddings.txt", embedding_type="combined")
