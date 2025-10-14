
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
                "Temporal CNNs":[],
                "State-space models":[],
            },
            "Graph and relational": [
                "Graph neural networks",
                "Graph transformers",
                "Relational reasoning networks",
                "Graph diffusion models",
                "Graph VAEs",
                "Graph GANs"
            ],
            "Physics-inspired and continuous models": [
                "Neural ODEs",
                "Hamiltonian neural networks",
                "Physics-informed neural networks"
            ],
            "Neuro-symbolic models": [
                "Neural-symbolic systems",
                "Differentiable logic layers"
            ],
        },
    },
    "AI problem type": {
        "Not relevant": [],
        "Natural language processing": [
            "Question answering",
            "Information extraction",
            "Semantic parsing",
            "Commonsense reasoning",
            "Machine translation",
            "Summarization",
            "Dialogue systems",
            "Human-AI co-writing",
            "Knowledge graph construction",
            "Grounded language understanding",
            "Information retrieval and search",
            "Retrieval-augmented generation"
        ],
        "Speech and audio": [
            "Speech recognition",
            "Speech synthesis",
            "Speaker diarization",
            "Audio event detection",
            "Music understanding",
            "Audio/music generation"
        ],
        "Computer vision": [
            "Image classification",
            "Video classification",
            "Object detection",
            "Segmentation",
            "Pose estimation",
            "Tracking",
            "3D reconstruction",
            "Scene understanding",
            "Simultaneous localization and mapping",
            "Text-to-image generation",
            "Image-to-image translation",
            "Video generation",
            "Neural rendering"
        ],
        "Multimodal": [
            "Vision-language understanding",
            "Audio-visual learning",
            "Cross-modal retrieval",
            "Multimodal generation"
        ],
        "Recommendation and personalization": [
            "Recommender systems",
            "Learning-to-rank",
            "Contextual bandits for recommendations"
        ],
        "Time-series and forecasting": [
            "Forecasting",
            "Anomaly detection",
            "Event prediction",
            "Demand and energy forecasting",
            "Financial forecasting"
        ],
        "Graph and relational": [
            "Node/edge/graph prediction",
            "Link prediction",
            "Community detection",
            "Graph clustering"
        ],
        "Decision-making and control": [
            "Planning and scheduling",
            "Robot manipulation",
            "Autonomous navigation",
            "Human-robot interaction",
            "Sim-to-real transfer"
            "Multi-agent decision-making",
            "Safe and robust control",
        ],
        "Structured and tabular": [
            "Classification",
            "Regression",
            "Survival analysis",
            "Risk scoring"
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
            "Ranking":[],
            "Structured prediction":[],
            "Sequence labeling":[],
            "Imbalanced learning":[],
            "Cost-sensitive learning":[]
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
            "Knowledge distillation": []
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
        "Training regimes": [
            "Curriculum learning",
            "Self-paced learning",
            "Hard example mining"
        ],
        "Meta-ML and model selection": [
            "Hyperparameter optimization",
            "AutoML (pipeline/search)",
            "Neural architecture search"
        ],

    },
    "Application domain": {
        "Not relevant": [],
        "General-purpose applications": {
            "Assistants and chat": {
                "Chatbots": [
                    "Open-domain/social chatbots",
                    "Task-oriented/transactional chatbots",
                    "Customer service chatbots",
                    "Domain-specific chatbots",
                    "Entertainment/roleplay chatbots",
                    "Research-oriented chatbots"
                ],
                "Customer support assistants":[],
                "Search assistants":[],
                "Personal assistants":[],
                "Knowledge management assistants":[],
                "General-purpose computer-use agents": [
                    "Browser automation agents",
                    "Desktop automation agents",
                    "Enterprise RPA-style agents",
                    "Developer/technical automation agents",
                    "Multi-app orchestration agents",
                    "Autonomous task loop agents"
                ],
            },
            "Productivity and office": [
                "Document summarization tools",
                "Meeting assistants",
                "Email drafting",
                "Note-taking and transcription",
                "Presentation/slide generation"
            ],
            "Developer and coding tools": [
                "Code completion",
                "Code explanation and refactoring",
                "Bug detection",
                "Test generation",
                "SQL/query assistants"
            ],
            "Creative tools": [
                "Image generation",
                "Video generation",
                "Music generation",
                "Story/poetry writing",
                "Design and prototyping"
            ],
            "Search and retrieval": [
                "Enterprise search",
                "Semantic search",
                "RAG-based assistants"
            ],
            "Analytics and decision support": [
                "Data analysis copilots",
                "Business intelligence augmentation",
                "Forecasting dashboards",
                "Simulation and scenario planning"
            ]
        },
        "Industry-specific applications": {
            "Healthcare and life sciences": {
                "Healthcare": [
                    "Clinical care and hospital operations",
                    "Medical imaging",
                    "Electronic health records",
                    "Public health and epidemiology"
                ],
                "Life sciences": [
                    "Drug discovery",
                    "Biotechnology",
                    "Genomics and proteomics"
                ]
            },
        
            "Finance, business and legal": {
                "Finance and fintech": [
                    "Banking and payments",
                    "Insurance",
                    "Investment and trading",
                    "Risk and compliance"
                ],
                "Legal and compliance": [
                    "Legal practice",
                    "E-discovery",
                    "Contract management",
                    "Regulatory compliance"
                ],
                "Business operations": [
                    "Human resources",
                    "Marketing and advertising",
                    "Customer service"
                ]
            },
        
            "Industry, manufacturing and infrastructure": {
                "Manufacturing": [
                    "Automotive",
                    "Electronics and hardware production",
                    "Chemicals and materials",
                    "Process industries"
                ],
                "Energy": {
                    "Oil and gas": [],
                    "Electricity": {
                        "Production": [
                            "Fossil",
                            "Nuclear",
                            "Solar",
                            "Wind",
                            "Hydro"
                        ],
                        "Grid and distribution": []
                    }
                },
                "Construction and real estate": [
                    "Architecture and design",
                    "Building operations",
                    "Real estate management"
                ],
                "Transportation and logistics": [
                    "Shipping and freight",
                    "Aviation",
                    "Rail",
                    "Autonomous vehicles"
                ],
                "Data centers and cloud computing": [
                    "Hyperscale cloud",
                    "Enterprise IT",
                    "Edge computing"
                ]
            },
        
            "Media, education and culture": {
                "Media and entertainment": [
                    "Film and video",
                    "Music",
                    "Publishing",
                    "Games"
                ],
                "Education": [
                    "Schools and universities",
                    "Online learning",
                    "Tutoring and assessment"
                ],
                "Culture and creative industries": [
                    "Art and design",
                    "Fashion",
                    "Heritage and museums"
                ]
            },
        
            "Technology platforms and ecosystems": {
                "AI and data platforms": [
                    "Foundation model providers",
                    "Model hosting and inference platforms",
                    "Vector databases and embeddings providers",
                    "MLOps and LLMOps platforms",
                    "Prompt/agent frameworks",
                    "Evaluation and observability",
                    "Synthetic data and labeling platforms",
                    "Feature stores and data platforms"
                ],
                "Open source ecosystems": [
                    "Model hubs and communities",
                    "Open model projects",
                    "Tooling and libraries",
                    "Open research resources"
                ],
                "Telecommunications": [
                    "Network infrastructure",
                    "Mobile operators",
                    "Satellite communications"
                ],
                "Mobile devices and ecosystems": [
                    "Smartphones",
                    "Wearables",
                    "IoT devices"
                ],
                "Operating systems and platforms": [
                    "Desktop OS",
                    "Mobile OS",
                    "Cloud platforms"
                ]
            },
        
            "Public sector, defense and security": {
                "Government": [
                    "Citizen services",
                    "Administration",
                    "Smart cities"
                ],
                "Defense and aerospace": [
                    "Military systems",
                    "Space exploration",
                    "Aerospace engineering"
                ],
                "Cybersecurity": [
                    "Threat detection",
                    "Incident response",
                    "Infrastructure protection"
                ]
            },
        
            "Agriculture and environment": {
                "Agriculture": [
                    "Crop production",
                    "Livestock",
                    "Precision agriculture"
                ],
                "Environment and sustainability": [
                    "Climate science",
                    "Conservation",
                    "Sustainable resource management"
                ]
            }
        }
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
        if self.current_path[-1] == "Other":
            raise ValueError
        result = find_child_nodes(self.TREE, self.current_path).copy()
        if self.shuffle:
            random.shuffle(result)
        if not "Not relevant" in result and len(result): # only add other option if there is no "not relevant" and its not already a leaf node
            result.append("Other")
        return result

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
    appl : list[str] = Field(description = "Application domain")

def get_v3_traverser_dict():
    traversers = {
            "arch": Traverser(CCS_HIERARCHY, ["Model architecture"]),
            "prob": Traverser(CCS_HIERARCHY, ["AI problem type"]),
            "para": Traverser(CCS_HIERARCHY, ["Learning paradigm"]),
            "appl": Traverser(CCS_HIERARCHY, ["Application domain"]),
            }
    return traversers


if __name__ == "__main__":
    t = Traverser(CCS_HIERARCHY, ["Model architecture"])
    t.set_traversal_type("down")
    
    print(t.get_child_nodes())
    t.move("Neural/deep learning architectures")
    print(t.get_child_nodes())

    t.reset_position(["AI problem type"])
    print(t.get_child_nodes())
    t.move("Applied domains")
    print(t.get_child_nodes())
    #print(t.get_sibling_nodes())
    #print(t.get_sibling_nodes())
    #print(t.get_parent_nodes())
    quit()
