import dspy
import pprint
import torch

import RAG
import restrictedmap
from chunking import chunk_by_headeres_and_clean
import keybert_ontology_mapping as kom




class ContextShortener():
    """ template for document shortener (RAG, rerank, etc) """
    def __init__(self):
        pass
    def set_document(self,document):
        self.document = document
    def __call__(self, **kwargs):
        raise NotImplementedError


class CreateRetrievalPromptSignature(dspy.Signature):
    # dspy signature (prompt template) for sequential form filling (i.e. one field at a time), field-agnistic.
    """
    You are a RAG prompt engineer working on retrieving specific details for filling out a form, using scientific papers as the documents.
    Make a retrieval prompt for finding the field described below
    """ 

    answer_field_name = dspy.InputField()
    answer_field_description = dspy.InputField()
    answer_field_examples = dspy.InputField()

    answer = dspy.OutputField(desc="String to be used for retrieving the above info from the context")

class FullPaperShortener(ContextShortener):
    """ output the whole document """
    def __call__(self, **kwargs):
        return self.document

class RAGShortener(ContextShortener):
    def __init__(self, chat_model, embed_model, pydantic_form, retriever_type, chunk_size, chunk_overlap, similarity_k, mmr_param):
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.pydantic_form = pydantic_form
        self.set_description_retrieval_prompt() # default : use description for retrieval
        self.retriever_type = retriever_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_k = similarity_k
        self.mmr_param = mmr_param

    def generate_retrieval_prompt_using_llm(self, dspy_lm):
        dspy.settings.configure(lm=dspy_lm)

        predictor = dspy.Predict(signature=CreateRetrievalPromptSignature)

        # iterate through fields
        fields = self.pydantic_form.__fields__
        retrieval_prompts = {}
        for fieldname in fields:
            field = fields[fieldname]
            
            retrieval_prompts[fieldname] = predictor(
                    answer_field_name = fieldname,
                    answer_field_examples = str(field.examples),
                    answer_field_description = field.description
                    ).answer

        print("\nretrieval prompts generated (Read through them and make sure they make sense!):")
        pprint.pprint(retrieval_prompts)
        self.retrieval_prompts = retrieval_prompts
        print("")

    def set_description_retrieval_prompt(self):
        # iterate through fields
        fields = self.pydantic_form.__fields__
        self.retrieval_prompts = {fieldname : fields[fieldname].description for fieldname in fields}
        print("retrieval prompts generated:")
        pprint.pprint(self.retrieval_prompts)

    def set_document(self,document):
        # make vectorstore
        vs = RAG.VectorStoreWeave(document=document,
                                  chat_model=self.chat_model,
                                  embed_model=self.embed_model,
                                  chunk_size = self.chunk_size,
                                  chunk_overlap = self.chunk_overlap,
                                  similarity_k = self.similarity_k,
                                  mmr_param = self.mmr_param,
                                  )

        # # Simple retriever
        # if self.retriever_type == "simple":
        #     self.retriever = vs.build_retriever()

        # # Fancy retriever
        # if self.retriever_type == "fusion":
        #     self.retriever = vs.build_fusion_retriever()

        # # Retriver with metadata (temporary)
        # if self.retriever_type == "metadata":
        #     self.retriever = vs.build_query_engine()

        self.retriever = vs.build_query_engine()


    def __call__(self, **kwargs):
        context_nodes = self.retriever.retrieve(self.retrieval_prompts[kwargs["answer_field_name"]])

        string_with_all_contexts = "\n...\n".join([node.get_text() for  node in context_nodes])
        return string_with_all_contexts
        #context = context_nodes[0].get_text()

        # print("\nRetrieved Context:")
        # print(context, "\n")

        #print("\n\nRetrieved Context Metadata:")
        #print(context_nodes[0].get_content(metadata_mode='all'))

        # TODO: rank context nodes by similarity match with filtered ontologies for relevant fields

        #return context


class Reduce(ContextShortener):
    def __init__(self, hf_model, hf_tokenizer, temperature, max_tokens, chunk_size, chunk_overlap, verbose=False):
        self.reducemodel = restrictedmap.RestrictedReduce(
                restrictedmap.RestrictedModel(
                    hf_model, 
                    hf_tokenizer, 
                    temperature = temperature,
                    max_new_tokens=max_tokens,
                    ),
                )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
    def set_document(self,document):
        self.reducemodel.set_chunks(
                restrictedmap.recursive_split(
                    document,
                    reverse=True,
                    chunk_size = self.chunk_size, # TODO move to init
                    chunk_overlap = self.chunk_overlap,
                    )
                )


    def __call__(self, **kwargs):
        reducemodel = self.reducemodel
        reducemodel.set_question(f"""Fill out the schema field with name '{kwargs["answer_field_name"]}' and description '{kwargs["answer_field_description"]}'.""")
        reducemodel.query_chunks()

        q = "Preliminary summaries:"

        for key in reducemodel.summaries:
            q += "\n---"+reducemodel.summaries[key]
        if self.verbose:
            print("")
            print(kwargs["answer_field_name"])
            print(q)
            print("")
        return q


class Rerank(ContextShortener):
    def __init__(self, hf_model, hf_tokenizer, verbose=False):
        self.rerankmodel = restrictedmap.RestrictedRerank(
                restrictedmap.RestrictedModel(
                    hf_model, 
                    hf_tokenizer, 
                    temperature = 0,
                    max_new_tokens=50
                    ),
                selection_mode = "combine",
                top_k = 5,
                restrict_scores = True)
        self.verbose = verbose
    def set_document(self,document):
        self.rerankmodel.set_chunks(
                restrictedmap.recursive_split(
                    document,
                    reverse=True,
                    chunk_size = 20000, # TODO move to init
                    chunk_overlap = 400,
                    )
                )


    def __call__(self, **kwargs):
        rerankmodel = self.rerankmodel
        rerankmodel.set_question(f"""Fill out the schema field with name '{kwargs["answer_field_name"]}' and description '{kwargs["answer_field_description"]}'.""")
        rerankmodel.query_chunks()
        rerankmodel.rank_chunks()

        q = "Preliminary answers:"
        top_k_scores = sorted(
                rerankmodel.scores, 
                key=rerankmodel.scores.get,
                reverse=True
                )[:min(rerankmodel.top_k, len(rerankmodel.scores))]
        for key in top_k_scores:
            q += "\n - " + rerankmodel.generated_answers[key]
        if self.verbose:
            print("")
            print(kwargs["answer_field_name"])
            print(q)
            print("")
        return q


class Keybert(ContextShortener):
    def __init__(self, mode, 
            n_keywords = 5, 
            top_k=3, 
            chunk_sizes = (5000,500),
            mmr_param = 1,
            maxsum_factor = 1,
            keyphrase_range = (1,1),
            ):
        self.n_keywords = n_keywords # number of keywords to extract from each chunk
        self.top_k = top_k # number of chunks to merge and return
        self.chunk_sizes = chunk_sizes
        assert mmr_param == 1 or maxsum_factor == 1
        self.mmr_param = mmr_param
        self.maxsum_factor = maxsum_factor
        self.keyphrase_range = keyphrase_range
        #self.mode = mode

        self.kw_model = kom.get_kw_model()
        self.emb_model = kom.get_embedding_model()
        

        self.descriptions = kom.get_subontology(mode) # TODO allow more and other fields
        self.target_emb = self.emb_model.encode(self.descriptions)

    def set_document(self, document):
        self.chunks = chunk_by_headeres_and_clean(document, chunk_size = self.chunk_sizes[0], chunk_overlap = self.chunk_sizes[1], verbose=False, split_by_periods=False)
        self.chunks = [chunk.text for chunk in self.chunks]

        self.keywordss = []
        self.keyword_scoress = []
        self.keyword_embeddingss = []
        self.indices_with_keywords = []
        for (i, chunk) in enumerate(self.chunks):
            keywords, scores = kom.get_keywords(
                    chunk, 
                    self.kw_model, 
                    # kwargs
                    keyphrase_ngram_range = self.keyphrase_range,
                    top_n=self.n_keywords,
                    use_maxsum = self.maxsum_factor>1,
                    use_mmr = self.mmr_param<1,
                    diversity = self.mmr_param,
                    nr_candidates = int(self.n_keywords * self.maxsum_factor),
                    )

            embs = self.emb_model.encode(keywords)
  
            if len(keywords)==0: # short chunks may have no keyword. Note that this require som extra index handling
                print(f"no keyword chunk: ***{chunk}***")
                continue
            self.keywordss.append(keywords)
            self.keyword_scoress.append(scores)
            self.keyword_embeddingss.append(embs)
            self.indices_with_keywords.append(i)

    def __call__(self, **kwargs):

        chunk_scores = []
        for kw_i, chunk_i in enumerate(self.indices_with_keywords): # keyword indices and chunk indices can be different

            similarity = kom.get_similarity_matrix(
                    self.keyword_embeddingss[kw_i], self.target_emb
                    )

            chunk_scores.append(
                    # store score and index
                    (self.calculate_chunk_relevance(similarity, self.keyword_scoress[kw_i]), chunk_i) # store index of corresponding chunk in tuple with the score
                    )

        chunk_scores = sorted(chunk_scores, key = lambda x: -x[0]) # sort in decreasing order, by score

        # print chunks
        #for score, index in chunk_scores:
        #    print(score, self.chunks[index])

        chosen_chunks = [self.chunks[chunk_scores[i][1]] for i in range(min(len(self.chunks), self.top_k))]
        return "\n...\n".join(chosen_chunks)



    def calculate_chunk_relevance(self, similarity, keyword_scores):

        # Reduce ontology value dimension NOTE this can be done in many different ways!
        #similarity_per_kw = similarity.mean(dim=1)
        similarity_per_kw = similarity.max(dim=1).values #  can also get indices, to get max description

        # reduce to single score # NOTE this can also be changed.
        keyword_scores = torch.Tensor(keyword_scores)
        product = similarity_per_kw.inner(keyword_scores)
        return product



if __name__=="__main__":
    text = "Monoclonal antibodies directed against cytotoxic T lymphocyte-associated antigen-4 (CTLA-4), such as ipilimumab, yield considerable clinical benefit for patients with metastatic melanoma by inhibiting immune checkpoint activity, but clinical predictors of response to these therapies remain incompletely characterized. To investigate the roles of tumor-specific neoantigens and alterations in the tumor microenvironment in the response to ipilimumab, we analyzed whole exomes from pretreatment melanoma tumor biopsies and matching germline tissue samples from 110 patients."
    kb = Keybert("dummy_mode", chunk_sizes = (100,0), keyphrase_range=(1,3))

    kb.descriptions = ["melanoma", "cancer", "sample"] # to speed up testing
    kb.target_emb = kb.emb_model.encode(kb.descriptions)

    kb.set_document(text)
    print(kb.keywordss)
    print(kb.keyword_scoress)
    print(kb())
