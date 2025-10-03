import random



CCS_HIERARCHY = {
    "Computing methodologies": {
        "Artificial intelligence": {
            "Foundation models": [
                "Transformer architectures",
                "Attention mechanisms",
                "Large language models",
                "Vision-language models",
                "Multimodal foundation models",
                "Retrieval-augmented generation",
                "Instruction tuning and alignment"
            ],
            "Generative AI": [
                "Diffusion models",
                "Generative adversarial networks",
                "Autoregressive generative models",
                "Variational autoencoders",
                "Neural rendering",
                "Text-to-image synthesis",
                "Text-to-audio and speech synthesis",
                "Co-creative systems"
            ],
            "Machine learning": {
                "Learning paradigms": {
                    "Supervised learning": [
                        "Few-shot and zero-shot learning",
                        "Self-supervised pretraining",
                        "Imbalanced learning",
                        "Structured prediction"
                    ],
                    "Unsupervised and self-supervised learning": [
                        "Representation learning",
                        "Contrastive learning",
                        "Clustering and density estimation",
                        "Matrix and tensor factorization"
                    ],
                    "Reinforcement learning": [
                        "Model-based reinforcement learning",
                        "Deep reinforcement learning",
                        "Offline reinforcement learning",
                        "Multi-agent reinforcement learning",
                        "Exploration strategies",
                        "Safe reinforcement learning"
                    ],
                    "Streaming and adaptive learning": [
                        "Online learning",
                        "Continual and lifelong learning",
                        "Meta-learning",
                        "Transfer learning"
                    ],
                    "Collaborative and privacy-aware learning": [
                        "Federated learning",
                        "Differential privacy in ML",
                        "Secure multi-party learning"
                    ]
                },
                "Model architectures": [
                    "Transformer models",
                    "Graph neural networks",
                    "Neural differential equation models",
                    "Probabilistic graphical models",
                    "Mixture-of-experts",
                    "Ensemble methods",
                    "Neural-symbolic systems"
                ],
                "Optimization and training": [
                    "Stochastic optimization",
                    "Second-order and adaptive methods",
                    "Curriculum learning",
                    "Parameter-efficient fine-tuning",
                    "Knowledge distillation",
                    "Regularization and sparsification",
                    "Hyperparameter optimization",
                    "AutoML"
                ],
                "Evaluation and validation": [
                    "Cross-validation",
                    "Robustness and adversarial testing",
                    "Uncertainty estimation",
                    "Benchmarking and reproducibility"
                ]
            },
            "Natural language processing": {
                "Core tasks": [
                    "Machine translation",
                    "Information extraction",
                    "Question answering",
                    "Dialogue systems",
                    "Summarization",
                    "Sentiment and affect analysis",
                    "Natural language generation",
                    "Speech recognition and synthesis"
                ],
                "Knowledge-enhanced NLP": [
                    "Grounded language understanding",
                    "Semantic parsing",
                    "Knowledge graph construction",
                    "Commonsense reasoning"
                ],
                "Multimodal and interactive NLP": [
                    "Vision-language understanding",
                    "Speech-language integration",
                    "Embodied conversational agents",
                    "Human-AI co-writing"
                ]
            },
            "Computer vision": {
                "Perception": [
                    "Image and video classification",
                    "Object detection and recognition",
                    "Segmentation and parsing",
                    "3D reconstruction",
                    "Scene understanding",
                    "Pose estimation",
                    "Tracking and motion analysis"
                ],
                "Generative vision": [
                    "Image-to-image translation",
                    "Neural rendering and view synthesis",
                    "Text-to-image generation",
                    "Video generation"
                ],
                "Spatial reasoning": [
                    "Simultaneous localization and mapping",
                    "Visual navigation",
                    "Geometric deep learning for vision"
                ]
            },
            "Multimodal AI": [
                "Cross-modal retrieval",
                "Multisensory fusion",
                "Vision-language models",
                "Audio-visual learning",
                "Embodied AI"
            ],
            "Robotics and control": [
                "Robot learning from demonstrations",
                "Vision-based control",
                "Manipulation and grasping",
                "Autonomous navigation",
                "Human-robot interaction",
                "Sim-to-real transfer"
            ],
            "Knowledge representation and reasoning": [
                "Neuro-symbolic reasoning",
                "Probabilistic reasoning",
                "Causal inference",
                "Explainable AI",
                "Temporal and sequential reasoning",
                "Spatial reasoning",
                "Knowledge graphs and ontologies"
            ],
            "Search and planning": [
                "Heuristic search",
                "Planning under uncertainty",
                "Hierarchical planning",
                "Monte Carlo tree search",
                "Learning-guided planning"
            ],
            "AI systems and engineering": [
                "MLOps and deployment",
                "Model compression and acceleration",
                "Edge and on-device AI",
                "Distributed training systems",
                "Data-centric AI"
            ],
            "Responsible and trustworthy AI": [
                "Fairness accountability and transparency",
                "Bias detection and mitigation",
                "AI safety and alignment",
                "Robustness and adversarial defenses",
                "Privacy-preserving AI",
                "Human-centered evaluation"
            ],
            "AI applications": [
                "Healthcare AI",
                "Scientific machine learning",
                "Climate and sustainability AI",
                "Creative and generative applications",
                "Autonomous systems",
                "Financial AI"
            ]
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

