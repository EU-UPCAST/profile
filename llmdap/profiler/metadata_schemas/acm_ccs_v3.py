
CCS_HIERARCHY = {
    "Model architecture": {
        "Not relevant": [],
        "Classical machine learning models": [
            "Linear regression",
            "Logistic regression",
            "Decision trees",
            "Random forests",
            "Gradient boosting",
            "Support vector machines"
        ],
        "Neural/deep learning architectures": {
            "Feed-forward networks": [
                "Multilayer perceptrons",
                "Autoencoders"
            ],
            "Convolutional networks": [
                "2D CNNs",
                "3D CNNs",
                "Convolutional autoencoders",
                "CNN-based GANs",
                "Diffusion UNets",
            ],
            "Sequence models": {
                "Recurrent neural networks": [
                    "Vanilla RNN",
                    "Long short-term memory (LSTM)",
                    "Gated recurrent unit (GRU)"
                ],
                "Transformer architectures": {
                    "Encoder-decoder transformers": [],
                    "Decoder-only transformers": [],
                    "Encoder-only transformers": [
                        "BERT",
                        "Vision transformers",
                        "Audio encoders"
                        ],
                    "Sparse attention transformers": [
                        "Long-context transformers",
                        "Low-rank / structured attention"
                    ],
                    "Mixture-of-experts transformers": [],
                    "Multimodal transformers": [
                        "Vision-language transformers",
                        "Speech-text transformers",
                        "Video-text transformers"
                    ],
                    "Variational transformers":[],
                    "Diffusion transformers":[],
                    "Adversarial transformers":[],
                },
                "Other sequence models": [
                    "Temporal CNNs",
                    "State-space models"
                ]
            },
            "Graph and relational": [
                "Graph neural networks",
                "Graph transformers",
                "Relational reasoning networks",
                "Graph diffusion models",
                "Graph VAEs",
                "Graph GANs"
            ],
            "Specialized hybrids": [
                "Mixture-of-experts",
                "Neural-symbolic systems",
                "Neural differential equation models",
                "Neural rendering pipelines"
            ]
        },
    },
    "AI problem": {
        "Not relevant": [],
        "Natural language processing": {
            "Core understanding": [
                "Question answering",
                "Information extraction",
                "Semantic parsing",
                "Commonsense reasoning"
            ],
            "Generation and interaction": [
                "Machine translation",
                "Summarization",
                "Dialogue systems",
                "Human-AI co-writing",
                "Speech recognition and synthesis"
            ],
            "Knowledge-enhanced": [
                "Knowledge graph construction",
                "Grounded language understanding"
            ]
        },
        "Computer vision": {
            "Perception": [
                "Image and video classification",
                "Object detection",
                "Segmentation",
                "Pose estimation",
                "Tracking"
            ],
            "3D and spatial": [
                "3D reconstruction",
                "Scene understanding",
                "Simultaneous localization and mapping"
            ],
            "Generative vision": [
                "Text-to-image generation",
                "Image-to-image translation",
                "Video generation",
                "Neural rendering"
            ]
        },
        "Multimodal and embodied": [
            "Vision-language understanding",
            "Audio-visual learning",
            "Cross-modal retrieval",
            "Embodied AI"
        ],
        "Decision-making and robotics": [
            "Robot manipulation",
            "Autonomous navigation",
            "Human-robot interaction",
            "Sim-to-real transfer"
        ],
        "Applied domains": [
            "Healthcare AI",
            "Scientific machine learning",
            "Climate and sustainability AI",
            "Financial AI",
            "Creative AI applications"
        ]
    },
    "Learning paradigm": {
        "Not relevant": [],
        "Supervised learning": {
            "Classification": [
                "Binary classification",
                "Multi-class classification",
                "Multi-label classification",
                "Hierarchical classification"
            ],
            "Regression": [
                "Single-output regression",
                "Multi-output regression",
                "Ordinal regression"
            ],
            "Other supervised tasks": [
                "Ranking",
                "Structured prediction",
                "Sequence labeling",
                "Imbalanced learning",
                "Cost-sensitive learning"
            ]
        },
        "Unsupervised learning": [
            "Contrastive learning",
            "Masked modeling",
            "Representation learning",
        ],
        "Self-supervised learning": [
            "Clustering",
            "Density estimation",
            "Dimensionality reduction",
        ],
        "Few-shot and generalization": [
            "Few-shot learning",
            "Zero-shot learning",
            "Prompt-based adaptation",
            "Meta-learning"
        ],
        "Model adaptation / fine-tuning": {
            "Full-parameter fine-tuning":[],
            "Task-specific fine-tuning":[],
            "Domain adaptation fine-tuning":[],
            "Continual fine-tuning":[],
            "Parameter-efficient tuning": [
                "Adapter-based tuning",
                "Low-rank adaptation (LoRA)",
                "Prefix / prompt tuning",
                "BitFit and bias-only tuning"
            ],
        },
        "Reinforcement learning": [
            "Model-based RL",
            "Model-free RL",
            "Offline RL",
            "Multi-agent RL",
            "Exploration strategies",
            "Safe RL"
        ],
        "Human feedback and alignment": [
            "Reinforcement learning from human feedback",
            "Constitutional AI",
            "Preference modeling",
            "Instruction tuning"
        ],
        "Collaborative and secure": [
            "Federated learning",
            "Differential privacy in ML",
            "Secure multi-party learning",
            "Continual and lifelong learning"
        ],
        "Automation and optimization": [
            "Hyperparameter optimization",
            "AutoML",
            "Curriculum learning",
            "Knowledge distillation"
        ]
    }
}


import random
from pydantic import BaseModel, Field, create_model
from typing import Literal

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
    def __init__(self, tree, start_path=[], shuffle_alternatives = True):
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


class v3_Schema(BaseModel):
    arch : list[str] = Field(description = "Model architecture")
    prob : list[str] = Field(description = "AI problem type")
    para : list[str] = Field(description = "Learning paradigm")

def get_v3_traverser_dict():
    traversers = {
            "arch": Traverser(CCS_HIERARCHY, ["Model architecture"]),
            "prob": Traverser(CCS_HIERARCHY, ["AI problem"]),
            "para": Traverser(CCS_HIERARCHY, ["Learning paradigm"])
            }
    return traversers


if __name__ == "__main__":
    t = Traverser(CCS_HIERARCHY, ["Model architecture"])
    t.set_traversal_type("down")
    
    print(t.get_child_nodes())
    t.move("Neural/deep learning architectures")
    print(t.get_child_nodes())

    t.reset_position(["AI problem"])
    print(t.get_child_nodes())
    t.move("Applied domains")
    print(t.get_child_nodes())
    #print(t.get_sibling_nodes())
    #print(t.get_sibling_nodes())
    #print(t.get_parent_nodes())
    quit()
