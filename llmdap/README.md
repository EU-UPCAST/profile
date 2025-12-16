





# UPCAST profiling

## trend analysis related files:
- `profiler/hf_trends.csv`: huggingface models, timestamps and predicted label.
- `profiler/arxiv_trencs.csv`: arxiv papers with predicted labels.
- `profiler/metadata_schemas/acm_ccs.py`: the acm-ccs taxonomy (with some changes explained at the top), + traverser class used for prediction.
- `profiler/trend_results.ipynb`: some initial visualizations


# old readme (for Biomed paper/dataset project):


## How to run:

setup environment:
```
conda create --name upcast python==3.10.14
conda activate upcast
pip install -r profiler/requirements.txt
pip install --upgrade werkzeug==2.3.8
```
(werkzeug must be downgraded from the required version, causing a warning)

run on a simple test case:
```
cd profiler
python main.py --dataset simple_test --no-log
```

Look through `introduction.ipynb` for an introduction to how the different parts work.

## Quick file overview

Folders:
- `data`: for preparing the datasets - downloading papers and metadata + collecting metadata into one json file
- `profiler`: the llm pipeline
- `profile/metadata_schemas`: pydantic schemas describing the format of the output of the pipeline
- `profiler/context_shortening/`: for reducing the size of the context fed to the llm (meant as a generalisation of the retrieval concept in rag - can be regular retrieval, keybert based retrieval, just feeding the full paper, or using llm to summarice)
- `profiler/form_filling/`: for the generation part, with structured output handled in different ways according to the llm and the flied. Returns an object following the correct pydantic schema for the specified dataset.


Some file descriptions:
#####- `profiler/main.py`: assemples the pipeline and runs it according to the arguments.
- `profiler/load_modules.py`: assemples the pipeline according to the arguments
- `profiler/run_modules.py`: runs the pipeline from load_modules.
- `profiler/dataset_loader.py` for dataset loading
- `profiler/arguments.yaml`: defines all the arguments for used in main.py and run_wandb_sweeps.py. Run `python main.py --help` to list them.
- `profiler/context_shortening/chunker_test.ipynb`: notebook with some testing of the chunker
- `profiler/optimize`: dspy prompt optimization - not updated in a while/currently not in use









## Experimental design


### Restricted prediction
- When generating answers for the field, we use restricted output.
  - For generation using llama3 or other open source models, Outlines is used to ensure the restrictions are followed. It works by setting the probabilities of disallowed tokens to 0 before sampling the generated tokens.
  This means, for integer fields the output will always be integer, for Literal fields it will always be one of the listed allowed answers, and for free-text it will follow the maximum length (which, when done this way instead of using max tokens, in most cases avoids halv sentences).
  - For generation through openai, we use their structured_output option, which works in the same way (masking disallowed tokens) but they have not implemented as many restrictions - e.g. they do not allow restricted field length, so this is ignored.
