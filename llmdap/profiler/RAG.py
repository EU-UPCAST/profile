import dspy
import weave
import metadata_schemas 
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from llama_index.core import get_response_synthesizer
from llama_index.core.schema import MetadataMode
from llama_index.core.schema import TextNode

# from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)

def set_openai_api_key():
    import openai
    from openai_key import API_KEY
    openai.api_key = API_KEY


class VectorStoreSimple():
    """ using LlamaIndex module """
    def __init__(self, document, llm):
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=32)
        Settings.text_splitter = text_splitter
        Settings.embed_model = OllamaEmbedding(model_name=llm)

        self.docs = Document(text=document)
        print('\ndocs type:', type(self.docs))

        self.chat_model = Ollama(model=llm)
        self.index = VectorStoreIndex.from_documents([self.docs], transformations=[text_splitter])


    def build_retriever(self, k=2):
        """ returns a list of "NodeWithScore" objects """

        return self.index.as_retriever(similarity_top_k = k)


    def build_query_engine(self):
        """ to use for direct querying for answers """

        return self.index.as_query_engine(llm=self.chat_model)



class VectorStoreWeave(weave.Model):
    document: str
    chat_model: str
    embed_model: str
    chunk_size: int = 2048
    chunk_overlap: int = 128
    similarity_k: int = 3
    mmr_param: float = 1.0
    index: VectorStoreIndex = None
    query_engine: RetrieverQueryEngine = None
    vector_retriever: VectorIndexRetriever = None

    # need to validate the signature scheme before initializing it here
    # signature: type 


    def _pseudo_markdown_splitter(self, text: str, chunk_size, chunk_overlap, markdown_headers=[]):
        """
        Borrowing the LangChain markdown splitter to spot header str in lieu of #, ##, ###, etc. 
        Keeps the md-formated header in medatdata dict if found, else nothing.
        Then recursively splits the text without breaking paragraphs.
        
        Args:
            text: str
                The text from UnstructuredXMLLoader in line-separated format header, subheader, pargaraphs.
            ...
            headers_to_split_on: list
                A list of tuples of the form (header, metadata_key) where header is a str that
                will be used to split the text and metadata_key is the key in the metadata dict
                that will be used to store ONLY markdown-formatted header.

        Returns:
            splits: list
                A list of Langchain doc objects, each containing paragraphs and artifact sub-headers according to 
                chunk size; and 'metadata' key would contain real markdown headers IF any.
        
        """

        from langchain_text_splitters import MarkdownHeaderTextSplitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Expands and relocate this as necessary
        markdown_headers = [
            ("METHODS", "methods"),
            ("RESULTS", "result"),
            ("FIG", "figure"),
            ("INTRO", "introduction"),
            ("REF", "reference"),
            ("SUPPL", "supplement"),
        ]

        # Split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(markdown_headers)
        md_header_splits = markdown_splitter.split_text(text)

        # Chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(md_header_splits)
        
        return splits


    def _match_sequence(self, seq1: str, seq2_list: list, threshold=0.7):
        """
        Gives a sequence similarity match between seq1 and seq2 from seq2_list.

        Args: 
            seq1: str
                The sequence to match
            seq_list: list
                A list of sequences to match against seq1
            threshold: float
                The minimum similarity ratio; <.8 for short sequences seems to work well.
        Return:
            seq2: str
                The sequence that passes the matching threshold or None if insufficient match.

        """

        from difflib import SequenceMatcher

        for seq2 in seq2_list:
            if SequenceMatcher(None, seq1, seq2).ratio() > threshold:
                return seq2
        return None


    def _split_non_md_headers(self, doc_lc, headers=[]):
        """ A secondery cleaner to extract leftover non-markdown headers from the text. 
        
        Args:
            doc_lc: LangChain Document 
                A list of Langchain doc objects, each containing paragraphs and artifact sub-headers according to 
                chunk size; and 'metadata' key would contain real markdown headers IF any.
            headers: list    
                A list of possible headers (e.g. headers = ["introduction", "paragraph", "title_1", "title_2", "fig_caption"])
        Return:
            cleaned_docs: dict
                Contains core paragraphs under dict['text'] and associated headers under dict['headers']

        """

        # Expands and relocate this as necessary
        headers = ["introduction",
                    "paragraph",
                    "title_1",
                    "title_2",
                    "fig_caption",
                    "abstract",
                    "supplementary material",
                    "materials and methods",
                    "results and discussion",
                    "footnote_title"
                    ]

        cleaned_docs = {}
        cleaned_docs['text'] = ""
        cleaned_docs['headers'] = []

        text = doc_lc.page_content

        for line in text.split("\n"):
            match = self._match_sequence(line.lower(), headers)
            if match:
                cleaned_docs['headers'].append(match)
            else:
                cleaned_docs['text'] += line

        return cleaned_docs


    def _store_nodes(self, document, verbose=True):
        """ indexing with metadata
        
        """
        
        # self._set_lm_models()
        extractors = self._set_metadata_extractors()
        pipeline = IngestionPipeline(transformations=extractors)

        nodes = []

        splits = self._pseudo_markdown_splitter(document, self.chunk_size, self.chunk_overlap)

        for doc in splits:
            split = self._split_non_md_headers(doc)  # headers currently defined within the function

            if verbose:
                print(f'\nMD Headers: {doc.metadata} \nOther Headers: {split["headers"]} \nSize change: {len(doc.page_content)} ? {len(split["text"])}')
                print(f'{split["text"]}')

            nodes.append(TextNode(text=split['text']))

        docs = pipeline.run(documents=nodes)

        self.index = VectorStoreIndex(nodes=docs)



    def _set_lm_models(self):
        if self.chat_model in ["text-embedding-3-small", "text-embedding-3-large"]:
            set_openai_api_key()
            #Settings.llm = OpenAI(model=self.chat_model) # TODO remove this
            Settings.embed_model = OpenAIEmbedding(model=self.chat_model)
        else:
            #Settings.llm = Ollama(model=self.chat_model) # TODO remove this
            Settings.embed_model = OllamaEmbedding(model_name=self.embed_model)


    def _store_text(self, document):
        """ indexing without metadata """
        self._set_lm_models()
        text_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.index = VectorStoreIndex.from_documents([Document(text=document)], transformations=[text_splitter])


    def build_retriever(self):
        """ returns a list of "NodeWithScore" objects """

        self._store_text(self.document)
        if self.mmr_param < 1:
            retriever = self.index.as_retriever(
                    vector_store_query_mode="mmr",
                    similarity_top_k = self.similarity_k,
                    vector_store_kwargs={"mmr_threshold": self.mmr_param},
                    )
        else:
            retriever = self.index.as_retriever(similarity_top_k = self.similarity_k)
        return retriever


    def _build_bm25_retriever(self):

        return BM25Retriever.from_defaults(docstore=self.index.docstore, similarity_top_k=self.similarity_k)


    def build_fusion_retriever(self):
        
        vector_retriever = self.build_retriever()
        bm25_retriever = self._build_bm25_retriever()

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self.similarity_k,
            num_queries=3,
            mode='reciprocal_rerank',
            use_async=False,
            verbose=True
        )
        return retriever


    def _set_metadata_extractors(self):
        """ define a set of mode metadata extractors for IngestionPipeline
            Note:   for input text from UnstructureXLMXMLLoader, exclude SentenceSplitter() as chunking
                    is performed at a prior stage.

        """

        # # Temporary LLM config
        # Settings.embed_model = OllamaEmbedding(model_name="llama3")
        # Settings.llm = Ollama(model="llama3") 

        metadata_extractors = [
            # SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            # QuestionsAnsweredExtractor(questions=3),
            # KeywordExtractor(keywords=5),
            # EntityExtractor(prediction_threshold=0.5)
        ]
        return metadata_extractors


    # def _store_nodes(self, document):
    #     """ indexing with metadata
    #         TODO: need to fix Doc loader for pipeline.run; temporary fix -> .from_documents()
    #     """
        
    #     self._set_lm_models()
    #     extractors = self._set_metadata_extractors()
    #     self.index = VectorStoreIndex.from_documents(
    #         [Document(text=document)], transformations=extractors)
        

    def build_query_engine(self):
        self._store_nodes(self.document)
        return self.index.as_retriever(similarity_top_k = self.similarity_k)


    def build_property_graph(self):
        pass


    @weave.op()
    def predict(self):
        pass
