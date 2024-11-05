import numpy as np
import torch
import transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_split(text, reverse=True, chunk_size=5000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(
            separators = ["\n\n", ".", ",", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            )
    texts_splitted = []
    if reverse:
        r = -1 # to have separatpr be at the end of a part, not the start of the next
    else:
        r=1
    text_splitted = text_splitter.split_text(text[::r])
    for i, part in enumerate(text_splitted[::r]):
        texts_splitted.append(part[::r])
    return texts_splitted



def listlist_to_tree(listlist, endtoken):
    tree = {}
    for inner_list in listlist:
        if not inner_list[0] in tree:
            tree[inner_list[0]] = []
        if len(inner_list) == 1:
            assert inner_list[0] == endtoken, (inner_list[0], endtoken)
        tree[inner_list[0]].append(inner_list[1:])

    for key in tree:
        if key != endtoken:
            tree[key] = listlist_to_tree(tree[key], endtoken)
        else:
            tree[key] = []
    return tree


class RestrictedModel:
    def __init__(self, model, tokenizer,temperature = 0.01, max_new_tokens=20, extra_eos_tokens=["\n", "\n\n", "\n\n\n", "\n\n\n\n", ".\n", ",\n", ".\n\n"]): # some of the most common tokens with \n. 
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=0, )
        self.max_new_tokens = max_new_tokens

        for token in extra_eos_tokens:
            assert len(self.tokenize(token)) == 1
        self.terminators = [
                    self.tokenizer.eos_token_id,
                    *[self.tokenize(token)[0] for token in extra_eos_tokens]
                    ]


        # check if there is a start-of-string token
        endtokenstring = "'"
        self.endtoken = self.tokenize(endtokenstring)
        if len(self.endtoken) == 1:
            self.endtoken = self.endtoken[0]
            self.has_start_token = 0
        elif len(self.endtoken) == 2: # is there a token at the start of every string=
            starttoken = self.endtoken[0]
            assert starttoken == self.tokenize('"')[0]
            assert starttoken == self.tokenize("a")[0]
            assert starttoken == self.tokenize("Me")[0]
            self.endtoken = self.endtoken[1]
            self.has_start_token = 1
        else:
            assert len(self.endtoken) == 1, f"End token length: len(self.endtoken)"
        print("has start token:", self.has_start_token)


    def tokenize(self, string):
        try:
            return self.tokenizer(string).data["input_ids"]
        except Exception as e:
            print("FAILED tokenizing string(s):", string)
            raise e
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def __call__(self, prompt, answers):
        torch.cuda.empty_cache()
        if answers is None:
            inputs = torch.tensor([self.tokenize(prompt)]).to(self.model.device)
            outputs = self.model.generate(inputs, 
                                          max_new_tokens=self.max_new_tokens,
                                          do_sample=True,
                                          top_k=50,
                                          top_p=0.95,
                                          eos_token_id=self.terminators,
                                          pad_token_id=self.terminators[0], # would be inferred and printed a msg
                                          )
            assert len(inputs) == 1 # the alternative is not implemented
            assert (outputs[0, :len(inputs[0])] == inputs[0]).all()
            outputs = outputs[:, len(inputs[0]):]
            output = self.tokenizer.batch_decode(outputs)[0]
            return output
        else:
            return self.make_restricted_call(prompt, answers)

        #print("clearing cache")
    
    def make_restricted_call(self, prompt, answers):

        endtoken = self.endtoken

        possible_tokens = self.tokenize(answers)
        possible_tokens = [ [*tokens[self.has_start_token:], endtoken] for tokens in possible_tokens]


        max_answer_length = max([len(inner_list) for inner_list in possible_tokens])

        possible_tokens = listlist_to_tree(possible_tokens, endtoken = endtoken)

        generation_config = transformers.GenerationConfig(
                    max_new_tokens=1, # amount of tokens, increase if you want more than one word output
                    eos_token_id=self.terminators,
                    pad_token_id=self.terminators[0], # would be inferred and printed a msg
                    do_sample=True,
                    temperature=self.temperature,
                    output_scores = True,
                    return_dict_in_generate=True,
                    )
       
        # tokenize prompt
        inputs = torch.tensor([self.tokenize(prompt)]).to(self.model.device)
        full_output = ""

        #endtokens = [endtoken, *self.terminators]

        for i in range(max_answer_length):
            if len(possible_tokens) > 1:

                # generate outputs
                outputs = self.model.generate(
                            inputs,
                            generation_config,
                            )
                outputs = outputs.scores[0][0]


                possible_keys = list(possible_tokens.keys())
                outputs = outputs[possible_keys]
                #temperature = 1000
                #probs = self.softmax(outputs/temperature)
                #print(probs)
                #print(outputs)

                argmax = torch.argmax(outputs)
                token = possible_keys[argmax] # TODO use softmax+temperature+ random sample instead of max # TODO for numbers, consider weighted avg instead of sampling - use high temperature.
            else:
                token = list(possible_tokens.keys())[0]
            
            #if token in endtokens:
            if token == endtoken:
                break
            output = self.decode(token)
            full_output += output

            # prepare for next iteration
            inputs = torch.cat((inputs, torch.tensor([[token]]).to(self.model.device)), dim = 1)
            possible_tokens = possible_tokens[token]

        return full_output

    def get_avg_score(self, prompt, minval, maxval):

        answers = [str(x) for x in range(minval,maxval+1)]
        possible_tokens = self.tokenize(answers)
        possible_tokens = [[*tokens[self.has_start_token:]] for tokens in possible_tokens]
        max_answer_length = max([len(inner_list) for inner_list in possible_tokens])
        assert max_answer_length == 1
        possible_tokens = [l[0] for l in possible_tokens]

        generation_config = transformers.GenerationConfig(
                    max_new_tokens=1,
                    eos_token_id=self.terminators,
                    pad_token_id=self.terminators[0], # would be inferred and printed a msg
                    do_sample=True,
                    output_scores = True,
                    return_dict_in_generate=True,
                    )

        # tokenize prompt
        inputs = torch.tensor([self.tokenize(prompt)]).to(self.model.device)

        # generate outputs
        outputs = self.model.generate(
                    inputs,
                    generation_config,
                    )
        outputs = outputs.scores[0][0]

        # restrict output
        #print(possible_tokens)
        outputs = outputs[possible_tokens]
        #print([self.decode(t) for t in possible_tokens])

        temperature = 100
        probs = self.softmax(outputs/temperature)
        #print(probs)
        #print(outputs)

        score = torch.inner( probs, torch.tensor(range(minval, maxval+1)).to(self.model.device)*1.0)
        #print(score)
        try: 
            score = round(score.item(),1)
        except ValueError as e:
            print("\n\n\n\n\n")
            print("printing probs and score")
            print(probs)
            print(score)
            print(outputs)
            score = int(minval)
            print(e)
        #print(score)
        return score


class RestrictedReduce:
    def __init__(self, restricted_model, allowed_answers=None):
        self.restricted_model = restricted_model
        self.allowed_answers = allowed_answers

    def set_chunks(self,chunks):
        if type(chunks) is dict:
            self.chunks = chunks
        elif type(chunks) is list:
            self.chunks = {i:chunks[i] for i in range(len(chunks))}
        else:
            raise NotImplementedError
        self.idx = list(self.chunks.keys())
    
    def set_question(self,question):
        self.question = question
    
    def __call__(self, ):
        
        self.query_chunks()
        answer = self.combine_answers()
        return answer
    
    def query_chunks(self):
        self.summaries = {}
        for i in self.idx:
            self.summaries[i] = self.restricted_model(self.query_chunk_prompt(i), answers=None)
        
    def combine_answers(self):
        return self.restricted_model(self.combine_prompt(), answers=self.allowed_answers)
    
    def query_chunk_prompt(self, i):
        q = f"""Summarize the following content based on the question provided below. That is, rewrite any infomation useful for answering the question, and ignore any irrelevant information.
Context: 
{self.chunks[i]}
(end of context)

Question: {self.question}

Summary: """
        return q

    def combine_prompt(self):
        q = f"""Following is a list of summaries from different parts of a research article.
Use this information to answer the question provided below as accurately as possible.

Summaries:
"""
        for key in self.summaries:
            q += self.summaries[key]
            q += """
"""
        q += """
Question:
{self.question}
"""
        if not self.allowed_answers is None:
            q += f"""The question is a multiple choice question. The answer must be one from the following list:
{self.allowed_answers}

Final answer: '"""
        else:
            q+= """Final answer: """
        return q

    def print_summaries(self):
        a = self.summaries
        for key in a.keys():
            print(a[key])


class RestrictedRerank:
    def __init__(self, restricted_model, allowed_answers = None, selection_mode = "best", top_k=5, restrict_scores=False):
        self.restricted_model = restricted_model
        assert selection_mode in ["best", "combine"]
        self.allowed_answers = allowed_answers
        self.selection_mode = selection_mode
        self.top_k = top_k # how many of the best answer to use when genereating the best (only with selection mode combine)
        self.restrict_scores = restrict_scores

    def set_chunks(self,chunks):
        if type(chunks) is dict:
            self.chunks = chunks
        elif type(chunks) is list:
            self.chunks = {i:chunks[i] for i in range(len(chunks))}
        else:
            raise NotImplementedError
        self.idx = list(self.chunks.keys())
    
    def set_question(self,question):
        self.question = question
    

    def __call__(self, ):
        
        self.query_chunks()
        self.rank_chunks()
        answer = self.get_best_answer()
        return answer
    
    def query_chunks(self):
        self.generated_answers = {}
        for i in self.idx:
            self.generated_answers[i] = self.restricted_model(self.query_chunk_prompt(i), answers = None)
        
    def rank_chunks(self):
        self.scores = {}
        #if self.restrict_scores:
        #    allowed_scores = [str(x) for x in range(1,99)]
        #else:
        #    allowed_scores = [str(x) for x in range(100)]

        for i in self.idx:
            #self.scores[i] = int(self.restricted_model(self.rank_prompt(i), answers = allowed_scores))
            self.scores[i] = self.restricted_model.get_avg_score(self.rank_prompt(i), minval=self.restrict_scores, maxval=100-self.restrict_scores)
        
    def get_best_answer(self):
        if self.selection_mode == "best":
            maxval = -1
            maxidx = 0
            for i in self.idx:
                if self.scores[i] > maxval:
                    maxval = self.scores[i]
                    maxidx = i
            return self.generated_answers[i]

        elif self.selection_mode == "combine":
            return self.restricted_model(self.combine_prompt(), answers =self.allowed_answers)
        raise NotImplementedError
    
    def query_chunk_prompt(self, i):
        q = f"""Use the following chunk of a document to answer the question provided below.
This answer will just be a preliminary answer, and answers from different pieces of a larger document will be used to produce the final answer afterwards.
Thus, if the context only contains part of the information required to answer the question, include this.
If no relevant information is found, do not come up with anything, instead just write "None".
"""

        if not self.allowed_answers is None:
            q += f"""Note that the final question is a multiple choice question, and the final answer must be one from the following list:
{self.allowed_answers}
However, for now any useful information is needed to get the correct answer later.
"""
        else:
            q += """
"""
        q += f"""
Context: {self.chunks[i]}
"""
        q+=f"""Question:
{self.question}

Preliminary answer: """
        return q

    def rank_prompt(self, i):
        q = f"""
Below is a question along with an answer. Your task is to score the answer: To what extent does the answer provide a satisfying answer to the question? 
Answer with a number from 0 to 100, where 0 means it contains no information relevant for answering the question, and 100 means it answers the question as precisely and in detail the question can be answered.

EXAMPLE:
Question: 'What day is it today?'
Answer: 'Monday'
Score: 38
The score in the example is relatively low since the answer does not contain date or year.

ACTUAL TASK:
Question: {self.question}
Answer: {self.generated_answers[i]}
Score: """
        return q

    def combine_prompt(self):
        q = f"""Following is a list of answers to a question, each based on different parts of a document that contain the true answer.
Use this information to answer the question provided below as accurately as possible.

Question:
{self.question}

Answers: 
"""
        top_k_scores = sorted(self.scores, key=self.scores.get, reverse=True)[:min(self.top_k, len(self.scores))]
        for key in top_k_scores:
            q += self.generated_answers[key]
            q += """
"""
        q += """
"""
        if not self.allowed_answers is None:
            q += f"""The question is a multiple choice question. The answer must be one from the following list:
{self.allowed_answers}

"""
        
        q+= """Final answer: '"""
        return q

    def print_answers(self):
        a = self.generated_answers
        s = self.scores
        sa = []
        for key in a.keys():
            sa.append((s[key],a[key]))
        sa = sorted(sa, key = lambda tup : tup[0])
        
        print("answers")
        for tupl in sa:
            print(tupl)




if __name__ == "__main__":



    def __get_chunks_for_testing():
        from langchain_community.document_loaders import UnstructuredXMLLoader
        import os
        
        loader = UnstructuredXMLLoader(
            "/mnt/data/upcast/data/all_xmls/18772890_ascii_pmcoa.xml",
            mode = "single", # "elements"
        )
        docs = loader.load()
        text = docs[0].page_content
        
        chunks = recursive_split(text)
        #for chunk in chunks:
        #    print(chunk)
        #    print("----@@@"*10)
    
        return chunks
    
    def test_reduce(): 
        chunks = __get_chunks_for_testing()
        answers = ["breast cancer", "lung cancer", "leukemia", "other"]

        model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
        device="cuda:0"
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map = device)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side = "left")

        reducemodel = RestrictedReduce(RestrictedModel(hf_model, hf_tokenizer), allowed_answers = answers)
        reducemodel.set_question("What cancer type is discussed in the paper?")
        reducemodel.set_chunks(chunks)
        
        out = reducemodel()
        
        print("Final output:")
        print(out)
        reducemodel.print_summaries()
        
        print("\n\n\nQUERY PROMPT")
        print(reducemodel.query_chunk_prompt(1))
        print("\n\n\nCOMBINE PROMPT")
        print(reducemodel.combine_prompt())
        
        
    def test_rerank():
        chunks = __get_chunks_for_testing()
        answers = ["breast cancer", "lung cancer", "leukemia", "other"]

        model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
        device="cuda:0"
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map = device)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side = "left")

        rerankmodel = RestrictedRerank(RestrictedModel(hf_model, hf_tokenizer), allowed_answers = answers, selection_mode = "combine", restrict_scores=True)
        rerankmodel.set_question("What cancer type is discussed in the paper?")
        rerankmodel.set_chunks(chunks)
        
        out = rerankmodel()
        
        print("Final output:")
        print(out)
        
        rerankmodel.print_answers()
        
        
        #print("\n\n\nQUERY PROMPT")
        #print(rerankmodel.query_chunk_prompt(1))
        #print("\n\n\nRANK PROMPT")
        #print(rerankmodel.rank_prompt(1))
        #print("\n\n\nCOMBINE PROMPT")
        #print(rerankmodel.combine_prompt())

    #test_reduce()
    test_rerank()
