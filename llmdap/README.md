# UPCAST_profiling



## Experimental design

### Dataset creation

#### Retrieving data
(this is done in the data/fetch_data.py file)
- Querying EuropePMC, we found all the ID of all the papers referred to by arrayExpress
- For each ID, we attempt downloading the full text XML of the paper from PMC, as well as the metadata from ArrayExpress.
- This results in 14860 metadata json files, for one dataset each, and the XMLs for their papers.
- These json files are quite big, include both information on the whole dataset and each sample, and which fields are in the json vary from dataset to dataset. Example: https://www.ebi.ac.uk/biostudies/files/E-MTAB-8097/E-MTAB-8097.json

#### Consolidating to a usable dataset
- (in data/make_somple_json.count_fields():) Iterate through the jsons, finding all unique fields, count the number of times they appear, note some examples and if they are ontology terms, and save this info
- Next, I disregarded any fields with less than 2000 appearances, and manually chose interesting candidates among the other, choosing only those 
  - with valuable information about the dataset (disregarding e.g. sample-specific fields)
  - that can likely be found in the paper (disregarding e.g. file names)
  - where there value is somewhat predictable (disregarding e.g. titles and descriptions).
  - NOTE: the majority of the fields were removed, leaving 23 candidates
  - NOTE: this step should perhaps be redone with more well-defined criteria?
- The 23 candidates are analysed in [this notebook](data/visualize_fields.ipynb)
  - Only fields with reasonably distributed values were selected (disregarding e.g. developmental stage and strain where most of the values are unique, and software, where most values are the same) 
  - This resulted in the fields found in the metadata schema.
  - Fields where all/most occurances have a small set of values, were set as Literal fields (i.e. multiple choice/categorization) (with values like "other" for values outside the selected allowable ones), the others as free text (still limited by length to avoid full sentece answers etc).
- (in the remainder of data/make_somple_json.count_fields()) The jsons were then merged into one json with the selected fields. In cases where a paper is connected to several datasets, all the values are used.

#### Further restrictions upon loading the data
- On the profiler pipeline, for now, all fields with more than one value is ignored.

### Restricted prediction
- When generating answers for the field, we use restricted output.
  - For generation using llama3 or other open source models, Outlines is used to ensure the restrictions are followed. It works by setting the probabilities of disallowed tokens to 0 before sampling the generated tokens.
  This means, for integer fields the output will always be integer, for Literal fields it will always be one of the listed allowed answers, and for free-text it will follow the maximum length (which, when done this way instead of using max tokens, in most cases avoids halv sentences).
  - For generation through openai, we use their structured_output option, which I am not sure how works under the hood. They do not allow restricted field length, so this is ignored.

### Evaluation
- Each field is evaluated in the following manner depending on the type:
  - Integer field: 1 if correct, 0 if not (independent of how far it was from correct)
  - Literal field: 1 if correct, 0 if not (with allowed answers based ontology we could introduce more levels, for values closeby in the tree)
  - string fields: Similarity based on characters
- Literal and integer reported as accuracy, strings as similarity.