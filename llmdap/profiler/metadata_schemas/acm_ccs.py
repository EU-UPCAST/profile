import random


# Changes to the taxnomomy:
# it is restricted to only AI and ML, 
# as well as some adjustments:
# Added other under computing methodologies (to not force non-AI papers into something its not
# moved machine learnging to Artificial intelligence (ML is part of AI)
# Merged machine learning algorithms with machine learning approaches (its the same thing)
# merged computer vision tasts with computer vision problems (its the same thing)


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
                    "Biometrics",
                    "Scene understanding",
                    "Activity recognition and understanding",
                    "Video summarization",
                    "Visual content-based indexing and retrieval",
                    "Visual inspection",
                    "Vision for robotics",
                    "Scene anomaly detection"
                ]
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
                    ],
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
            },
        },
        "Other": []
    }
}



from pydantic import BaseModel, Field, create_model
from typing import Literal

class Downward_Traverser:
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
        self.current_path = ["Computing methodologies"] # TODO generalize to other ontologies

def find_child_nodes(tree, path):
    subtree = tree
    for key in path:
        if type(subtree) is list: # if subtree is a list, and there is still a key in path, the key is a list element, i.e. leaf node (there are no child nodes)
            return []
        subtree = subtree[key]
    if type(subtree) is dict:
        return list(subtree.keys())
    else:
        assert type(subtree) is list, (subtree, path, type(subtree))
        return subtree

class Traverser:
    def __init__(self, tree, start_path=["Computing methodologies"], shuffle_alternatives = True):
        self.TREE = tree
        self.start_path = start_path
        self.reset_position(start_path)
        self.shuffle = shuffle_alternatives

    def set_traversal_type(self,traversal_type):
        if traversal_type == "down":
            self.include_parents = False
            self.include_siblings = False
        elif traversal_type == "vertical":
            self.include_parents = True
            self.include_siblings = False
        elif traversal_type == "free":
            self.include_parents = True
            self.include_siblings = True
        else:
            raise ValueError


    def get_child_nodes(self):
        result = find_child_nodes(self.TREE, self.current_path)
        if self.shuffle:
            random.shuffle(result)
        return  result
    def get_sibling_nodes(self):
        siblings = find_child_nodes(self.TREE, self.current_path[:-1]).copy()
        siblings.remove(self.current_path[-1])
        return siblings
    def get_parent_nodes(self):
        # NOTE: for now, we assume single parent
        if len(self.current_path) < 2:
            return []
        return [self.current_path[-2]]

    def move(self, new_node):
        if new_node in self.get_child_nodes():
            self.current_path.append(new_node)
            return "v"
        elif new_node in self.get_parent_nodes() and self.include_parents:
            self.current_path = self.current_path[:-1]
            return "^"
        elif new_node in self.get_sibling_nodes() and self.include_siblings:
            self.current_path = self.current_path[:-1] + [new_node]
            return "<"
        else:
            raise ValueError # New node is not in the alternatives


    def get_pydantic_form(self):
        possible_values= self.get_child_nodes() + [self.current_path[-1]]
        if self.include_parents:
            possible_values.extend(self.get_parent_nodes())
        if self.include_siblings:
            possible_values.extend(self.get_sibling_nodes())
        assert len(possible_values) == len(set(possible_values)), possible_values # Checks that there is no duplicate. otherwise llm output is ambiguous
        if len(possible_values) == 1: # no reason to keep generating if there is only one possibility
            assert possible_values[0] == self.current_path[-1]
            raise StopIteration
        if self.shuffle:
            random.shuffle(possible_values)

        field_type = Literal[tuple(possible_values)]# make Literal dynamically by converting to tuple
    
        fieldname = "field_"+self.current_path[-1].replace(" ","_").replace("-","_").replace(",", "_").replace("/","_")
        field = Field(description = "Most relevant subnode")
        pydantic_form = create_model(fieldname, **{"next_part_of_path":(field_type, field)})
        return pydantic_form
    
    def get_field(self):
        pydantic_form = self.get_pydantic_form()
        return next(iter(pydantic_form.model_fields.values()))

    def reset_position(self, path=None):
        if path is None:
            self.current_path = self.start_path.copy()
        else:
            self.current_path = path.copy()


class PathSchema(BaseModel):
    path : list[str] = Field(description = "path to node position in tree")

if __name__ == "__main__":
    t = Traverser(CCS_HIERARCHY)

    t.set_traversal_type("free")
    t.move("Artificial intelligence")
    t.move("Knowledge representation and reasoning")
    t.move("Vagueness and fuzzy logic")
    print(t.get_child_nodes())
    print(t.get_sibling_nodes())
    print(t.get_sibling_nodes())
    print(t.get_parent_nodes())
    quit()

