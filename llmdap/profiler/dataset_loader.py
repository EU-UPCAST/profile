import json
from langchain_community.document_loaders import UnstructuredXMLLoader


def get_simple_test(max_amount):
    """ Simple test made to check that things work in a quick/low cost way/simple way """

    abstract1 = """In this paper we analyse 45 biosamples from brain tumors of a female brown rat. (James P. Salot, University of Copenhagen, 2015)"""
    papers = {
            #"0":abstract1,
            #"1":abstract1,
            #"2":abstract1,
            #"3":abstract1,
            "4":abstract1,
            }

    labels = {
        #    "0":{
        #"sex_2": ["female"], #literal 
        #},
        #    "1":{
        #"releasedate_12" : [2015],# int
        #},
        #    "2":{
        #"organism_16": ["rattus norvegicus"]# constr(40)
        #},
        #    "3":{
        #"name_19 ": ["univerity of copenhagen"]# constr, need description to understand this is organisation name, not author name
        #},
            "4":{
        "sex_2": ["female"], #literal 
        "releasedate_12" : [2015],# int
        "sample_count_13": [45],# int
        "organism_16": ["rattus norvegicus"],# constr(40)
        "name_19": ["univerity of copenhagen"]# constr, need description to understand this is organisation name, not author name
        }}


    # add emtpy arrays (i.e. missing labels) for the other fields in the arxpr schema
    import metadata_schemas 
    for fieldname in metadata_schemas.arxpr_schema.__fields__:
        for label_id in labels:
            if not fieldname in labels[label_id]:
                labels[label_id][fieldname] = []

    return papers, labels

def load_nhrf_examples2(max_amount):
    data_folder = "/mnt/data/upcast/data/"
    labels = {
            "26359337":{
                "dataset_design": ["Melanoma vs germ-line"],
                "organism": ["Human", "Homo sapiens"],
                "sample_type": ["FFPE tissue"],
                "sequencing_technology": ["Whole Exome sequencing", "RNA sequencing"],
                "phenotypic_state": ["Matching germ-line tissue", "Melanoma: Cutaneous", "Occult", "Mucosal"],
                "tissue": ["Melanoma", "matching germline tissue"],
                "N_sample_conditions": [4],
                "dataset_size": [120, 40], # 120 WES and 40 RNAseq 
                "experimental_factor": ["Histological Subtype "],
                "experimental_factor_value": ["Matching germ-line tissue", "Melanoma: Cutaneous", "Occult", "Mucosal"],
                "raw": ["yes", "private", "dbGAP"],
                "processed": ["yes", "public", "cbioportal"],
                }
            }
    paper_texts, labels = load_paper_text(labels, max_amount, data_folder)
    return paper_texts, labels

def load_nhrf_examples3(max_amount):
    paper_texts, labels = load_nhrf_examples2(max_amount)
    return paper_texts, {}

def load_nhrf_examples(max_amount):
    data_folder = "/mnt/data/upcast/data/"

    dummy_labels = {
            33495476:0,
            35810190:0,
            33789117:0,
            35368039:0,
            }
    paper_texts, labels = load_paper_text(dummy_labels, max_amount, data_folder)
    return paper_texts, {}

def load_ega_data(max_amount = 10):
    """ mode : "single" or "elements" """
    data_folder = "/mnt/data/upcast/data/"

    with open(data_folder + "ega/prepared_dataset.json") as file:
        labels = json.load(file)

    paper_texts, labels = load_paper_text(labels, max_amount, data_folder)
    return paper_texts, labels

def load_arxpr_data(max_amount = 10, version = "", mode = "train"):
    """ mode : "single" or "elements" """
    data_folder = "/mnt/data/upcast/data/"

    if mode == "train":
        with open(data_folder + f"arxpr{version}_metadataset_train.json") as file:
            labels = json.load(file)
    elif mode == "test":
        with open(data_folder + f"arxpr{version}_metadataset_holdout.json") as file:
            labels = json.load(file)

    ## count fields:
    #items = list(labels.items())
    #ones = {field:0 for field in items[0][1]}
    #anys = {field:0 for field in items[0][1]}

    #for i in range(min(len(labels), max_amount)):
    #    for field in items[i][1]:
    #        l = len(items[i][1][field])
    #        if l>0:
    #            anys[field] += 1
    #        if l==1:
    #            ones[field] += 1
    #from pprint import pprint
    #print("N datasets with exactly one label, for each field:")
    #pprint(ones)
    #print("N datasets with at least one label, for each field")
    #pprint(anys)
    #quit()


    paper_texts, labels = load_paper_text(labels, max_amount, data_folder)

    return paper_texts, labels

class Arxpr_generator:
    def __init__(self, version = "", mode = "train"):
        """ mode : "single" or "elements" """
        self.data_folder = "/mnt/data/upcast/data/"

        if mode == "train":
            with open(self.data_folder + f"arxpr{version}_metadataset_train.json") as file:
                self.labels = json.load(file)
        elif mode == "test":
            with open(self.data_folder + f"arxpr{version}_metadataset_holdout.json") as file:
                self.labels = json.load(file)

        self.i = 0
        self.keys = list(self.labels.keys())

    def get_next_labels(self):
        if self.i >= len(self.labels):
            return None
        key = self.keys[self.i]
        self.i += 1
        return key, self.labels[key]

    def get_paper_text(self, key):
        paper_texts, labels = load_paper_text({key:self.labels[key]}, 1, self.data_folder)
        assert len(paper_texts) == len(labels)
        if len(paper_texts) == 0:
            return None
        assert len(paper_texts) == 1
        return paper_texts[key]

class Studytype_generator(Arxpr_generator):
    def get_next_labels(self):
        if self.i >= len(self.labels):
            return None
        key = self.keys[self.i]
        self.i += 1
        return key, {"study_type_18": self.labels[key]["study_type_18"] if "study_type_18" in self.labels[key] else []}


def load_study_type_data(max_amount = 10):
    """ mode : "single" or "elements" """
    data_folder = "/mnt/data/upcast/data/"

    with open(data_folder + "arxpr_metadataset_train.json") as file:
        train_labels = json.load(file)
    #with open(data_folder + "arxpr_metadataset_holdout.json") as file:
    #    holdout_labels = json.load(file)

    # restrict labels to those with study type, and remove the other fields
    study_type_labels = {}
    type_name = "study_type_18"
    for i, key in enumerate(train_labels):
        if i>= max_amount:
            break
        if type_name in train_labels[key]:
            study_type_labels[key] = {type_name : train_labels[key][type_name]}
    train_labels = study_type_labels

    train_paper_texts, train_labels = load_paper_text(train_labels, max_amount, data_folder)
    ###holdout_paper_texts, holdout_labels = load_paper_text(holdout_labels, max_amount)

    return train_paper_texts, train_labels


def load_paper_text(labels, max_amount,data_folder, mode = "elements"):

    full_xmls = {}
    i = 0
    for key in labels:
        try: # TODO move this tp its own fnc to avoid repeating
            xml_file = data_folder + f"all_xmls/{key}_ascii_pmcoa.xml"

            # single
            if mode == "single":
                full_xmls[key] = UnstructuredXMLLoader(xml_file, mode = "single").load()[0].page_content
        
            elif mode == "elements":
                # element
                docs = UnstructuredXMLLoader(xml_file, mode = "elements").load()
                string = ""
                for doc in docs:
                    # ignore useless metadata, + some 
                    if doc.metadata["category"] != "UncategorizedText":
                        string += doc.page_content + "\n"
                full_xmls[key] = string
            else:
                raise ValueError

            i+=1
            #print(f"loading, {i}/{max_amount}")
            if i>= max_amount:
                break
        except FileNotFoundError:
           continue
    # only include labels for the xmls included
    labels = {key:labels[key] for key in full_xmls}

    return full_xmls, labels

if __name__=="__main__":
    # print(load_ega_data(1)[1])
    t,l = load_arxpr_data(5, "2_25")
    print(l.keys())
    generator = Arxpr_generator("2_25")
    for i in range(5):
        print(generator.get_next_labels()[0])
