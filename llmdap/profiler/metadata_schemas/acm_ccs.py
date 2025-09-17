
CCS_HIERARCHY = {
    "Computing methodologies": {
        "Artificial intelligence": {
            "Natural language processing": [
                "Information extraction",
                "Machine translation",
                "Discourse, dialogue and pragmatics",
                "Natural language generation",
                "Speech recognition",
                "Lexical semantics",
                "Phonology / morphology",
                "Language resources"
            ],
            "Knowledge representation and reasoning": [
                "Description logics",
                "Semantic networks",
                "Nonmonotonic, default reasoning and belief revision",
                "Probabilistic reasoning",
                "Vagueness and fuzzy logic",
                "Causal reasoning and diagnostics",
                "Temporal reasoning",
                "Cognitive robotics",
                "Ontology engineering",
                "Logic programming and answer set programming",
                "Spatial and physical reasoning",
                "Reasoning about belief and knowledge"
            ],
            "Planning and scheduling": [
                "Planning for deterministic actions",
                "Planning under uncertainty",
                "Multi-agent planning",
                "Planning with abstraction and generalization",
                "Robotic planning",
                "Evolutionary robotics"
            ],
            "Search methodologies": [
                "Heuristic function construction",
                "Discrete space search",
                "Continuous space search",
                "Randomized search",
                "Game tree search",
                "Abstraction and micro-operators",
                "Search with partial observations"
            ],
            "Control methods": [
                "Robotic planning",
                "Evolutionary robotics",
                "Computational control theory",
                "Motion path planning"
            ],
            "Philosophical/theoretical foundations of artificial intelligence": [
                "Cognitive science",
                "Theory of mind"
            ],
            "Distributed artificial intelligence": [
                "Multi-agent systems",
                "Intelligent agents",
                "Mobile agents",
                "Cooperation and coordination"
            ],
            "Computer vision": {
                "Computer vision tasks": [
                    "Biometrics",
                    "Scene understanding",
                    "Activity recognition and understanding",
                    "Video summarization",
                    "Visual content-based indexing and retrieval",
                    "Visual inspection",
                    "Vision for robotics",
                    "Scene anomaly detection"
                ],
                "Image and video acquisition": [
                    "Camera calibration",
                    "Epipolar geometry",
                    "Computational photography",
                    "Hyperspectral imaging",
                    "Motion capture",
                    "3D imaging",
                    "Active vision"
                ],
                "Computer vision representations": [
                    "Image representations",
                    "Shape representations",
                    "Appearance and texture representations",
                    "Hierarchical representations"
                ],
                "Computer vision problems": [
                    "Interest point and salient region detections",
                    "Image segmentation",
                    "Video segmentation",
                    "Shape inference",
                    "Object detection",
                    "Object recognition",
                    "Object identification",
                    "Tracking",
                    "Reconstruction",
                    "Matching"
                ]
            }
        },
        "Machine learning": {
            "Learning paradigms": {
                "Supervised learning": [
                    "Ranking",
                    "Learning to rank",
                    "Supervised learning by classification",
                    "Supervised learning by regression",
                    "Structured outputs",
                    "Cost-sensitive learning"
                ],
                "Unsupervised learning": [
                    "Cluster analysis",
                    "Anomaly detection",
                    "Mixture modeling",
                    "Topic modeling",
                    "Source separation",
                    "Motif discovery",
                    "Dimensionality reduction and manifold learning"
                ],
                "Reinforcement learning": [
                    "Sequential decision making",
                    "Inverse reinforcement learning",
                    "Apprenticeship learning",
                    "Multi-agent reinforcement learning",
                    "Adversarial learning"
                ],
                "Multi-task learning": [
                    "Transfer learning",
                    "Lifelong machine learning",
                    "Learning under covariate shift"
                ]
            },
            "Learning settings": [
                "Batch learning",
                "Online learning settings",
                "Learning from demonstrations",
                "Learning from critiques",
                "Learning from implicit feedback",
                "Active learning settings",
                "Semi-supervised learning settings"
            ],
            "Machine learning approaches": {
                "Classification and regression trees": [],
                "Kernel methods": [
                    "Support vector machines",
                    "Gaussian processes"
                ],
                "Neural networks": [],
                "Logical and relational learning": [
                    "Inductive logic learning",
                    "Statistical relational learning"
                ],
                "Learning in probabilistic graphical models": [
                    "Maximum likelihood modeling",
                    "Maximum entropy modeling",
                    "Maximum a posteriori modeling",
                    "Mixture models",
                    "Latent variable models",
                    "Bayesian network models"
                ],
                "Learning linear models": [
                    "Perceptron algorithm"
                ],
                "Factorization methods": [
                    "Non-negative matrix factorization",
                    "Factor analysis",
                    "Principal component analysis",
                    "Canonical correlation analysis",
                    "Latent Dirichlet allocation"
                ],
                "Rule learning": [],
                "Instance-based learning": [],
                "Markov decision processes": [],
                "Partially-observable Markov decision processes": [],
                "Stochastic games": [],
                "Learning latent representations": [
                    "Deep belief networks"
                ],
                "Bio-inspired approaches": [
                    "Artificial life",
                    "Evolvable hardware",
                    "Genetic algorithms",
                    "Genetic programming",
                    "Evolutionary robotics",
                    "Generative and developmental approaches"
                ]
            },
            "Machine learning algorithms": {
                "Dynamic programming for Markov decision processes": [
                    "Value iteration",
                    "Q-learning",
                    "Policy iteration",
                    "Temporal difference learning",
                    "Approximate dynamic programming methods"
                ],
                "Ensemble methods": [
                    "Boosting",
                    "Bagging"
                ],
                "Spectral methods": [],
                "Feature selection": [],
                "Regularization": []
            },
            "Cross-validation": []
        }
    }
}


from pydantic import BaseModel, Field, create_model
from typing import Literal

class Traverser:
    def __init__(self, tree):
        self.TREE = tree
        self.reset()

    def get_child_nodes(self):
        path = self.current_path
        subtree = self.TREE 
        for key in path:
            if type(subtree) is list: # if subtree is a list, and there is still a key in path, the key is a list element, i.e. leaf node (there are no child nodes)
                return []
            subtree = subtree[key]
        if type(subtree) is dict:
            return list(subtree.keys())
        else:
            assert type(subtree) is list, (subtree, path, type(subtree))
            return subtree
    
    def is_leaf_node(self):
        path = self.current_path
        return len(self.get_child_nodes()) == 0
    
    def get_next_pydantic_form(self):
        path = self.current_path
        if self.is_leaf_node():
            raise StopIteration
        possible_values = [path[-1], *self.get_child_nodes()] # also add parent node, used for stopping traversal
        field_type = Literal[tuple(possible_values)]# make Literal dynamically by converting to tuple
    
        fieldname = "field_"+path[-1].replace(" ","_").replace("-","_").replace(",", "_").replace("/","_")
        field = Field(description = "Most relevant subnode")
        pydantic_form = create_model(fieldname, **{"next_part_of_path":(field_type, field)})
        return pydantic_form
    
    def get_next_field(self):
        pydantic_form = self.get_next_pydantic_form()
        return next(iter(pydantic_form.model_fields.values()))

    def set_next_step(self, key):
        if key == self.current_path[-1]:
            raise StopIteration
        assert key in self.get_child_nodes()
        self.current_path.append(key)

    def reset(self,):
        self.current_path = ["Computing methodologies"]


class PathSchema(BaseModel):
    path : list[str] = Field(description = "path to node position in tree")

if __name__ == "__main__":
    t = Traverser()
    f = t.get_next_field()
    #f = f.model_fields

    print(f)
    print(type(f))
    print(f.annotation)
