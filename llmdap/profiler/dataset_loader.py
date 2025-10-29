import json
import glob

from langchain_community.document_loaders import UnstructuredXMLLoader


def get_simple_test(max_amount):
    """ Simple test made to check that things work in a quick/low cost way/simple way """

    abstract1 = """In this paper we analyse 45 biosamples from brain tumors of a female brown rat. (James P. Salot, University of Copenhagen, 2015)"""
    papers = {
            "4":abstract1,
            }

    labels = {
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
    """ test of single paper with NHRF ground truth """
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
    """ get the ega (European Genome-Phenome Archive) dataset """
    data_folder = "/mnt/data/upcast/data/"

    with open(data_folder + "ega/prepared_dataset.json") as file:
        labels = json.load(file)

    paper_texts, labels = load_paper_text(labels, max_amount, data_folder)
    return paper_texts, labels

def load_arxpr_data(max_amount = 10, version = "", mode = "train"):
    """ load arrayepress dataset 

    version: "" or "2_25", or "3_X. Version 2 has fewer fields (more carefully picked) with only some labels included (25).
    3_25_X is normalised (as good as possible. X is number of values per field. Dataset must be made first (for each new X)"""
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
    """ similar to load_arxpr_data, but in (pseudo-)generator styele - documents are loaded one at a time as needed, instead of upfront """
    def __init__(self, version = "", mode = "train"):
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
            raise StopIteration
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
    """ like Arxpr_generator but with only the study type labels (for using ontology information) """
    def get_next_labels(self):
        if self.i >= len(self.labels):
            return None
        key = self.keys[self.i]
        self.i += 1
        return key, {"study_type_18": self.labels[key]["study_type_18"] if "study_type_18" in self.labels[key] else []}

def load_arxiv_papers(max_amount=10, full_text=False):
    if full_text:
        raise NotImplementedError
    import pandas as pd
    filepath = "/mnt/data/upcast/data/arxiv_ai_taxonomy_papers.csv"
    df = pd.read_csv(filepath, index_col=0)
    abstracts = df["abstract"].to_dict()
    titles= df["title"].to_dict()
    papers = {}
    i = 0
    for key in titles:
        papers[str(key)] = f"Title: {titles[key]}\nAbstract: {abstracts[key]}"
        i += 1
        if i>= max_amount:
            break

    return papers, None


def load_study_type_data(max_amount = 10):
    """ like load_arxpr_data but with only the study type labels (for using ontology information) """
    data_folder = "/mnt/data/upcast/data/"

    with open(data_folder + "arxpr_metadataset_train.json") as file:
        train_labels = json.load(file)

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

    return train_paper_texts, train_labels


def load_paper_text(labels, max_amount,data_folder, mode = "elements"):
    """ 
    Given labels dict, loads the paper texts using the keys (pmids).
    Also removes any labels not used (due to missing papers, or max_amount reached).

    mode : "single" or "elements" """

    full_xmls = {}
    i = 0
    for key in labels:
        try:
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

def load_paper_text_from_file_path(xml_file, mode = "elements"):
    """ 
    loads singe xml file from absolute path (used in inference)
    mode : "single" or "elements" """
    if mode == "single":
        # single
        paper_text = UnstructuredXMLLoader(xml_file, mode = "single").load()[0].page_content
    
    elif mode == "elements":
        # element
        docs = UnstructuredXMLLoader(xml_file, mode = "elements").load()
        string = ""
        for doc in docs:
            # ignore useless metadata, + some 
            if doc.metadata["category"] != "UncategorizedText":
                string += doc.page_content + "\n"
        paper_text = string
    else:
        raise ValueError
    return paper_text

def load_paper_text_from_url(paper_url):
    import requests
    page = requests.get(paper_url)
    raw_text = page.text
    return parse_raw_xml_string(raw_text)

def parse_raw_xml_string(raw_xml_string):
    from unstructured.partition.xml import partition_xml
    from lxml.etree import XMLSyntaxError
    try:
        docs = partition_xml(text=raw_xml_string)
    except XMLSyntaxError as e:
        print("The xml string was not parsable as xml")
        raise e

    string = ""
    for doc in docs:
        if doc.category != "UncategorizedText":
            string += doc.text+ "\n"
    return string

def count_fields(test_or_train = "train"):
    g = Arxpr_generator("2_25", test_or_train)

    train_numbers = {
            "papers" : 0,
            'hardware_4': 0,
            'organism_part_5': 0,
            'experimental_designs_10': 0,
            'assay_by_molecule_14': 0,
            'study_type_18': 0
            }

    i =0
    while True:

        if i%10==0:
            print(i) # progress
        i += 1

        try:
            key, l = g.get_next_labels()
        except:
            break
        text = g.get_paper_text(key)
        if not text is None:
            assert len(text)>50, text
            if len(text)>50:
                train_numbers["papers"] += 1
                for k in l:
                    train_numbers[k] += len(l[k])
            else:
                print("short", key, text) #never happens

            #print(key)
            #print(l)
        else:
            print("missin", key) #never happens

    print(train_numbers)
    # for train: {'papers': 5000, 'hardware_4': 349, 'organism_part_5': 600, 'experimental_designs_10': 546, 'assay_by_molecule_14': 4452, 'study_type_18': 4247}
    # for test:  {'papers': 6371, 'hardware_4': 483, 'organism_part_5': 821, 'experimental_designs_10': 707, 'assay_by_molecule_14': 5672, 'study_type_18': 5444}





def download_latest_arxiv_data():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    return path

def update_arxiv_file():
    path = download_latest_arxiv_data()
    path += "/arxiv-metadata-oai-snapshot.json"

    arxiv_data = []  # create empty list
    for line in open(path, 'r'):  # open file line by line
        arxiv_data.append(json.loads(line))  # append data to the initially created list

    df = pd.DataFrame.from_records(arxiv_data)  # convert data to dataframe
    df = df.set_index("id")  # set unique id as index

    # Drop Irrelevant features
    df = df.drop(['submitter', 'authors', 'comments', 'journal-ref', 'doi', 'report-no', 'license', 'update_date'], axis=1)

    # Filter by category
    category_list = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.MA', 'cs.NE', 'cs.RO', 'stat.ML']
    df = df[df['categories'].isin(category_list)]

    # Rename the columns to be engineered
    df.rename(columns={'versions': 'submission_date'}, inplace=True)


    # Convert dictionary items in submission_date to a list and access the second element
    df['submission_date'] = df['submission_date'].apply(
        lambda x: x[0]['created'] if isinstance(x, list) and len(x) > 0 and 'created' in x[0] else None)

    df["submission_date"] = pd.to_datetime(df["submission_date"])
    df.to_csv("/mnt/data/upcast/data/arxiv_ai_taxonomy_papers.csv")

def update_HF_dataset():
    from datasets import load_dataset
    ds = load_dataset("librarian-bots/model_cards_with_metadata")
    ds.save_to_disk("/mnt/data/upcast/data/trend_analysis/model_cards_with_metadata/")


def _load_hf_timeline(hf_data_path: str = "/mnt/data/upcast/data/trend_analysis/model_cards_with_metadata/train") -> "pd.DataFrame":
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc


    files = sorted(glob.glob(f"{hf_data_path}/data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow shards found in {hf_data_path}")

    tables = []
    for path in files:
        with open(path, "rb") as stream:
            reader = ipc.open_stream(stream)
            tables.append(reader.read_all())

    table = pa.concat_tables(tables)
    df = table.select(["modelId", "createdAt", "last_modified", "card", "tags", "pipeline_tag"]).to_pandas()

    df.loc[df["card"].str.len() == 5171, "card"] = "" # remove autogenerated model cards

    # restrict to minimum 500 char total
    minimum_char_requirement = 500
    total = df["card"].str.len() + df["tags"].str.len()+ df["pipeline_tag"].str.len().fillna(0) # only pipeline_tag can be NaN
    df = df[total>=minimum_char_requirement]

    df["createdAt"] = pd.to_datetime(df["createdAt"]).dt.tz_convert(None)
    df["last_modified"] = pd.to_datetime(df["last_modified"]).dt.tz_convert(None)
    return df


def _load_arxiv_timeline(arxiv_csv_path: str = "/mnt/data/upcast/data/arxiv_ai_taxonomy_papers.csv") -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(arxiv_csv_path, index_col=0, low_memory=False)
    df["submission_date"] = pd.to_datetime(df["submission_date"], utc=True).dt.tz_convert(None)
    return df


def _load_newsletter_timelines(csv_paths  = [
            "/mnt/data/upcast/data/import_ai_stories.csv",
            "/mnt/data/upcast/data/tldr_ai_stories.csv",
            "/mnt/data/upcast/data/dlweekly_stories.csv"
            ]):
    import pandas as pd

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        dfs.append(df)
    return dfs



def find_longest_true_sebseq(series, debug = False):
    # find longest consecutive subsequence of True

    # count position in subseq
    s2 = series.copy()
    s2 = s2.astype(int)
    count=0
    for id in series.index:
        if series.loc[id]:
            count +=1
        else:
            count = 0
        s2.loc[id] = count

    if debug:
        return s2
    series = series.iloc[:s2.argmax()+1] # remove part after the longest subseq
    series = series.iloc[-series[::-1].argmin():] # remove part before
    assert all(series)

    return series


class Longterm_Datasets:
    def __init__(self, period=[2013.0, 2022.02]):
        self.full_arx = _load_arxiv_timeline()
        full_nls = _load_newsletter_timelines()
        self.full_imp = full_nls[0]
        self.full_dlw = full_nls[2]

        self.period = period

    def prepare(self, m=1):
        self.arx = self.full_arx.copy()
        self.dlw = self.full_dlw.copy()
        self.imp = self.full_imp.copy()

        self.group_dfs_in_months(m)

        # restrict to range
        self.arx = self.arx[self.arx["bin"]>=self.period[0]][self.arx["bin"]<=self.period[1]]
        self.dlw = self.dlw[self.dlw["bin"]>=self.period[0]][self.dlw["bin"]<=self.period[1]]
        self.imp = self.imp[self.imp["bin"]>=self.period[0]][self.imp["bin"]<=self.period[1]]

    def sample_subsets(self, n):
        RANDOM_STATE = 123
        arxsubset=self.arx.groupby(["bin"]).sample(n=n, random_state = RANDOM_STATE)
        dlwsubset=self.dlw.groupby(["bin"]).sample(n=n, random_state = RANDOM_STATE)
        impsubset=self.imp.groupby(["bin"]).sample(n=n, random_state = RANDOM_STATE)
        return arxsubset, dlwsubset, impsubset

    def get_dict_format(self, n):
        arx, dlw, imp = self.sample_subsets(n)

        abstracts = arx["abstract"].to_dict()
        titles= arx["title"].to_dict()
        papers = {}
        for key in titles:
            papers[str(key)] = f"Title: {titles[key]}\nAbstract: {abstracts[key]}"

        newsletters = []
        for nl in dlw:
            newsletters.append(nl["text"].to_dict())
        for nl in imp:
            newsletters.append(nl["text"].to_dict())

        return papers, newsletters

    def group_dfs_in_months(self, m = 1): #m= number of months to combine in a bin (e.g. m=3 means we look at quarters)
        self.arx["bin"] = ((self.arx["submission_date"].dt.month+m-1)//m)/100 + self.arx["submission_date"].dt.year
        self.dlw["bin"] = ((self.dlw["submission_date"].dt.month+m-1)//m)/100 + self.dlw["submission_date"].dt.year
        self.imp["bin"] = ((self.imp["submission_date"].dt.month+m-1)//m)/100 + self.imp["submission_date"].dt.year


class Arxiv_HF_Newsletters_datasets:
    def __init__(self):
        self.full_arx = _load_arxiv_timeline()
        self.full_hf = _load_hf_timeline()
        self.full_nls = _load_newsletter_timelines()

    def prepare(self, m=1, threshold=1103):
        self.threshold = threshold
        self.arx = self.full_arx.copy()
        self.hf = self.full_hf.copy()
        self.nls = [nl.copy() for nl in self.full_nls]

        self.group_dfs_in_months(m)

        # find longest range of groups with plenty of examples. This only considers arxiv and HF
        period = self.find_commonly_plentiful_subseries(threshold)
        period = find_longest_true_sebseq(period)
        bin_range = period.index.min(), period.index.max()

        # restrict to range
        self.arx = self.arx[self.arx["bin"]>=bin_range[0]][self.arx["bin"]<=bin_range[1]]
        self.hf = self.hf[self.hf["bin"]>=bin_range[0]][self.hf["bin"]<=bin_range[1]]
        restricted_nls = []
        for nl in self.nls:
            nl = nl[nl["bin"]>=bin_range[0]][nl["bin"]<=bin_range[1]]
            restricted_nls.append(nl)
        self.nls = restricted_nls

    def sample_subsets(self, n):
        n = min(n, self.threshold)
        RANDOM_STATE = 123

        hfsubset=self.hf.groupby(["bin"]).sample(n=n, random_state = RANDOM_STATE)
        arxsubset=self.arx.groupby(["bin"]).sample(n=n, random_state = RANDOM_STATE)

        nl_subsets = []
        for nl in self.nls:
            min_group = nl.groupby(["bin"]).count()["text"].min()
            nl = nl.groupby(["bin"]).sample(n=min(n,min_group), random_state = RANDOM_STATE)
            nl_subsets.append(nl)

        return hfsubset, arxsubset, nl_subsets

    def get_dict_format(self, n):
        hf, arx, nl_subsets= self.sample_subsets(n)

        abstracts = arx["abstract"].to_dict()
        titles= arx["title"].to_dict()
        papers = {}
        for key in titles:
            papers[str(key)] = f"Title: {titles[key]}\nAbstract: {abstracts[key]}"

        card = hf["card"].to_dict()
        tags = hf["tags"].to_dict()
        ptag = hf["pipeline_tag"].to_dict()
        mid = hf["modelId"].to_dict()
        hfmodels = {}
        for key in card:
            hfmodels[mid[key].replace("/","__")] = f"ModelId: {mid[key]}\n\nTags: {tags[key]}\n\npipeline_tag: {ptag[key]}\n\nModel card:\n{card[key]}"

        newsletters = []
        for nl in nl_subsets:
            newsletters.append(nl["text"].to_dict())

        return hfmodels, papers, newsletters


    def find_commonly_plentiful_subseries(self, threshold = 2000):

        arxiv_plentiful_months = self.arx.groupby(["bin"]).count()["title"] >= threshold
        hf_plentiful_months = self.hf.groupby(["bin"]).count()["modelId"] >= threshold

        both_plentify_months = arxiv_plentiful_months & hf_plentiful_months

        start = both_plentify_months[both_plentify_months==True].index.min()
        end = both_plentify_months[both_plentify_months==True].index.max()

        period = both_plentify_months.loc[start:end]

        return period

    def group_dfs_in_months(self, m = 1): #m= number of months to combine in a bin (e.g. m=3 means we look at quarters)
        self.arx["bin"] = ((self.arx["submission_date"].dt.month+m-1)//m)/100 + self.arx["submission_date"].dt.year
        self.hf["bin"] = ((self.hf["createdAt"].dt.month+m-1)//m)/100 + self.hf["createdAt"].dt.year
        for nl in self.nls:
            nl.loc[:,"bin"] = ((nl["date"].dt.month+m-1)//m)/100 + nl["date"].dt.year

class Arxiv_HF_datasets:
    def __init__(self):
        self.full_arx = _load_arxiv_timeline()
        self.full_hf = _load_hf_timeline()

    def prepare(self, m=1, threshold=1103):
        self.threshold = threshold
        self.arx = self.full_arx.copy()
        self.hf = self.full_hf.copy()

        self.group_dfs_in_months(m)

        # find longest range of groups with plenty of examples
        period = self.find_commonly_plentiful_subseries(threshold)
        period = find_longest_true_sebseq(period)
        bin_range = period.index.min(), period.index.max()

        # restrict to range
        self.arx = self.arx[self.arx["bin"]>=bin_range[0]][self.arx["bin"]<=bin_range[1]]
        self.hf = self.hf[self.hf["bin"]>=bin_range[0]][self.hf["bin"]<=bin_range[1]]

    def sample_subsets(self, n):
        n = min(n, self.threshold)
        RANDOM_STATE = 123

        hfsubset=self.hf.groupby(["bin"]).sample(n=n, random_state = RANDOM_STATE)
        arxsubset=self.arx.groupby(["bin"]).sample(n=n, random_state = RANDOM_STATE)

        return hfsubset, arxsubset

    def get_dict_format(self, n):
        hf, arx = self.sample_subsets(n)

        abstracts = arx["abstract"].to_dict()
        titles= arx["title"].to_dict()
        papers = {}
        for key in titles:
            papers[str(key)] = f"Title: {titles[key]}\nAbstract: {abstracts[key]}"

        card = hf["card"].to_dict()
        tags = hf["tags"].to_dict()
        ptag = hf["pipeline_tag"].to_dict()
        mid = hf["modelId"].to_dict()
        hfmodels = {}
        for key in card:
            hfmodels[mid[key].replace("/","__")] = f"ModelId: {mid[key]}\n\nTags: {tags[key]}\n\npipeline_tag: {ptag[key]}\n\nModel card:\n{card[key]}"

        return hfmodels, papers


    def find_commonly_plentiful_subseries(self, threshold = 2000):

        arxiv_plentiful_months = self.arx.groupby(["bin"]).count()["title"] >= threshold
        hf_plentiful_months = self.hf.groupby(["bin"]).count()["modelId"] >= threshold

        both_plentify_months = arxiv_plentiful_months & hf_plentiful_months

        start = both_plentify_months[both_plentify_months==True].index.min()
        end = both_plentify_months[both_plentify_months==True].index.max()

        period = both_plentify_months.loc[start:end]

        return period

    def group_dfs_in_months(self, m = 1): #m= number of months to combine in a bin (e.g. m=3 means we look at quarters)
        self.arx["bin"] = ((self.arx["submission_date"].dt.month+m-1)//m)/100 + self.arx["submission_date"].dt.year
        self.hf["bin"] = ((self.hf["createdAt"].dt.month+m-1)//m)/100 + self.hf["createdAt"].dt.year



def test_arxhf_sampler():
    ahs = Arxiv_HF_datasets()
    ahs.prepare()
    hf3, arx3 = ahs.sample_subsets(3)
    hf, arx = ahs.sample_subsets(1103)
    del ahs
    ahs = Arxiv_HF_datasets()
    ahs.prepare()
    hf1, arx1 = ahs.sample_subsets(1)

    assert all(hf.groupby(["bin"]).count()["modelId"]==1103)
    assert all(hf3.groupby(["bin"]).count()["modelId"]==3)
    assert all(arx.groupby(["bin"]).count()["title"]==1103)

    assert hf1.index.isin(hf3.index).all()
    assert hf3.index.isin(hf.index).all()
    assert arx1.index.isin(arx3.index).all()
    assert arx3.index.isin(arx.index).all()

    assert len(arx) == 1103*43, (len(arx), len(arx1))
    assert len(hf3) == 3*43


    print("tests passed")




if __name__ == "__main__":
    test_arxhf_sampler()
