from __future__ import annotations
from .Log import Logger
import itertools
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import Counter
import langchain_core.runnables
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance
from datetime import datetime, timezone

Logger = Logger()


# Get the current year
current_year = datetime.now().year

# Create a datetime object for January 1st of the current year at midnight in UTC
january_first = datetime(current_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Format the datetime object in ISO 8601 format
iso_format = january_first.isoformat()

# import nltk
#
# # Download necessary resources from NLTK
# nltk.download('punkt')

if TYPE_CHECKING:
    from azure.cosmos.cosmos_client import CosmosClient


# def keyword_query(text, k):
#     tokens = nltk.word_tokenize(text)
#     stopwords = nltk.corpus.stopwords.words('english')
#     filtered_tokens = [token for token in tokens if token not in stopwords]
#     filtered_tokens = filtered_tokens[::-1]
#     where_clause = ""
#     for keyword in filtered_tokens:
#         where_clause += f" Lower(c.text) LIKE Lower('%{keyword}%') OR"
#     where_clause = where_clause[:-3]
#     query = f"""SELECT Top {k} c.id, c.text, c.source, c.category FROM c WHERE {where_clause}"""
#     return query

class AzureCosmosDBNoSqlVectorSearch(VectorStore):
    """`Azure Cosmos DB for NoSQL` vector store.

    To use, you should have both:
        - the ``azure-cosmos`` python package installed

    You can read more about vector search using AzureCosmosDBNoSQL here:
    https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
    """

    def __init__(
            self,
            *,
            cosmos_client: CosmosClient,
            embedding: Embeddings,
            vector_embedding_policy: Dict[str, Any],
            indexing_policy: Dict[str, Any],
            cosmos_container_properties: Dict[str, Any],
            cosmos_database_properties: Dict[str, Any],
            database_name: str = "vectorSearchDB",
            container_name: str = "vectorSearchContainer",
            create_container: bool = True,
    ):
        """
        Constructor for AzureCosmosDBNoSqlVectorSearch

        Args:
            cosmos_client: Client used to connect to azure cosmosdb no sql account.
            database_name: Name of the database to be created.
            container_name: Name of the container to be created.
            embedding: Text embedding model to use.
            vector_embedding_policy: Vector Embedding Policy for the container.
            indexing_policy: Indexing Policy for the container.
            cosmos_container_properties: Container Properties for the container.
            cosmos_database_properties: Database Properties for the container.
        """
        self._cosmos_client = cosmos_client
        self._database_name = database_name
        self._container_name = container_name
        self._embedding = embedding
        self._vector_embedding_policy = vector_embedding_policy
        self._indexing_policy = indexing_policy
        self._cosmos_container_properties = cosmos_container_properties
        self._cosmos_database_properties = cosmos_database_properties
        self._create_container = create_container

        if self._create_container:
            if (
                    indexing_policy["vectorIndexes"] is None
                    or len(indexing_policy["vectorIndexes"]) == 0
            ):
                raise ValueError(
                    "vectorIndexes cannot be null or empty in the indexing_policy."
                )
            if (
                    vector_embedding_policy is None
                    or len(vector_embedding_policy["vectorEmbeddings"]) == 0
            ):
                raise ValueError(
                    "vectorEmbeddings cannot be null "
                    "or empty in the vector_embedding_policy."
                )
            if self._cosmos_container_properties["partition_key"] is None:
                raise ValueError(
                    "partition_key cannot be null or empty for a container."
                )

        # Create the database if it already doesn't exist
        self._database = self._cosmos_client.create_database_if_not_exists(
            id=self._database_name,
            offer_throughput=self._cosmos_database_properties.get("offer_throughput"),
            session_token=self._cosmos_database_properties.get("session_token"),
            initial_headers=self._cosmos_database_properties.get("initial_headers"),
            etag=self._cosmos_database_properties.get("etag"),
            match_condition=self._cosmos_database_properties.get("match_condition"),
        )

        # Create the collection if it already doesn't exist
        self._container = self._database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=self._cosmos_container_properties["partition_key"],
            indexing_policy=self._indexing_policy,
            default_ttl=self._cosmos_container_properties.get("default_ttl"),
            offer_throughput=self._cosmos_container_properties.get("offer_throughput"),
            unique_key_policy=self._cosmos_container_properties.get(
                "unique_key_policy"
            ),
            conflict_resolution_policy=self._cosmos_container_properties.get(
                "conflict_resolution_policy"
            ),
            analytical_storage_ttl=self._cosmos_container_properties.get(
                "analytical_storage_ttl"
            ),
            computed_properties=self._cosmos_container_properties.get(
                "computed_properties"
            ),
            etag=self._cosmos_container_properties.get("etag"),
            match_condition=self._cosmos_container_properties.get("match_condition"),
            session_token=self._cosmos_container_properties.get("session_token"),
            initial_headers=self._cosmos_container_properties.get("initial_headers"),
            vector_embedding_policy=self._vector_embedding_policy,
        )

        self._embedding_key = self._vector_embedding_policy["vectorEmbeddings"][0][
                                  "path"
                              ][1:]

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        _metadatas = list(metadatas if metadatas is not None else ({} for _ in texts))

        return self._insert_texts(list(texts), _metadatas)

    def _insert_texts(
            self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Used to Load Documents into the collection

        Args:
            texts: The list of documents strings to load
            metadatas: The list of metadata objects associated with each document

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # If the texts is empty, throw an error
        if not texts:
            raise Exception("Texts can not be null or empty")

        # Embed and create the documents
        embeddings = self._embedding.embed_documents(texts)
        text_key = "text"

        to_insert = [
            {"id": str(uuid.uuid4()), text_key: t, self._embedding_key: embedding, **m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in CosmosDB No Sql
        doc_ids: List[str] = []
        for item in to_insert:
            created_doc = self._container.create_item(item)
            doc_ids.append(created_doc["id"])
        return doc_ids

    @classmethod
    def _from_kwargs(
            cls,
            embedding: Embeddings,
            *,
            cosmos_client: CosmosClient,
            vector_embedding_policy: Dict[str, Any],
            indexing_policy: Dict[str, Any],
            cosmos_container_properties: Dict[str, Any],
            cosmos_database_properties: Dict[str, Any],
            database_name: str = "vectorSearchDB",
            container_name: str = "vectorSearchContainer",
            **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        if kwargs:
            warnings.warn(
                "Method 'from_texts' of AzureCosmosDBNoSql vector "
                "store invoked with "
                f"unsupported arguments "
                f"({', '.join(sorted(kwargs))}), "
                "which will be ignored."
            )

        return cls(
            embedding=embedding,
            cosmos_client=cosmos_client,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            database_name=database_name,
            container_name=container_name,
        )

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        """Create an AzureCosmosDBNoSqlVectorSearch vectorstore from raw texts.

        Args:
            texts: the texts to insert.
            embedding: the embedding function to use in the store.
            metadatas: metadata dicts for the texts.
            **kwargs: you can pass any argument that you would
                to :meth:`~add_texts` and/or to the 'AstraDB' constructor
                (see these methods for details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an `AzureCosmosDBNoSqlVectorSearch` vectorstore.
        """
        vectorstore = AzureCosmosDBNoSqlVectorSearch._from_kwargs(embedding, **kwargs)
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            self._container.delete_item(document_id)
        return True

    def delete_document_by_id(self, document_id: Optional[str] = None) -> None:
        """Removes a Specific Document by id

        Args:
            document_id: The document identifier
        """
        if document_id is None:
            raise ValueError("No document ids provided to delete.")
        self._container.delete_item(document_id, partition_key=document_id)

    def _similarity_search_with_score(
            self,
            user_query,
            embeddings: List[float],
            k: int = 4,
    ) -> List[Tuple[Document, float]]:
        print("user_query",user_query)
        user_query_list = user_query.split()
        if len(user_query_list) == 1:
            print("single user_query")
            query = (
                    "SELECT TOP {} c.id, c.text,c.source,c.category,c.Date,VectorDistance(c.{}, {}) AS "
                    "SimilarityScore FROM c WHERE lower(c.text) like '%{}%' ORDER BY VectorDistance(c.{}, {}) ".format(
                        k,
                        self._embedding_key,
                        embeddings,
                        user_query,
                        self._embedding_key,
                        embeddings,
                    )
                )
            #Logger.log(f"query-"+query, "Info")

        else:
            if "holiday" in user_query:
                print("holiday_query")
                query = (
                    "SELECT TOP {} c.id, c.text,c.source,c.category,c.Date,VectorDistance(c.{}, {}) AS "
                    "SimilarityScore FROM c WHERE c.source like '%Holiday_Calendar%' and c.category='PageMenu' ORDER BY VectorDistance(c.{}, {})".format(
                        k,
                        self._embedding_key,
                        embeddings,
                        self._embedding_key,
                        embeddings,
                    )
                )
            else:
                query = (
                    "SELECT TOP {} c.id, c.text,c.source,c.category,c.Date,VectorDistance(c.{}, {}) AS "
                    "SimilarityScore FROM c ORDER BY VectorDistance(c.{}, {}) ".format(
                        k,
                        self._embedding_key,
                        embeddings,
                        self._embedding_key,
                        embeddings,
                    )
                )
        docs_and_scores = []
        items1 = list(
            self._container.query_items(query=query, enable_cross_partition_query=True)
        )
        source = ''
        if items1:
            source = items1[0]["source"]
            print(source)
            category = items1[0]["category"]
            if category == "News" or category == "Banner":
                latest_date = None
                for item in items1:
                    date = item["Date"]
                    date_obj = datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")
                    if latest_date is None or date_obj > latest_date:
                        latest_date = date_obj
                        source = item["source"]
        if "C:" in source:
            source = source.replace("\\", "\\\\")
        if "D:" in source:
            source = source.replace("\\", "\\\\")
        if "holiday" in user_query:
            query2 = (
                "SELECT TOP 3 c.id, c.text,c.source,c.category,VectorDistance(c.{}, {}) AS "
                "SimilarityScore FROM c WHERE c.source = '{}' and c.category='PageMenu' ORDER BY VectorDistance(c.{}, {})".format(
                    self._embedding_key,
                    embeddings,
                    source,
                    self._embedding_key,
                    embeddings,
                )
            )
        else:
            query2 = (
                "SELECT TOP 3 c.id, c.text,c.source,c.category,VectorDistance(c.{}, {}) AS "
                "SimilarityScore FROM c WHERE c.source = '{}' ORDER BY VectorDistance(c.{}, {})".format(
                    self._embedding_key,
                    embeddings,
                    source,
                    self._embedding_key,
                    embeddings,
                )
            )
            
        items2 = list(
            self._container.query_items(query=query2, enable_cross_partition_query=True)
        )
        for item in items2:
            text = item["text"]
            print("\n\n")
            print(item["source"])
            print(text)
            text = str(text).replace("\n", "")
            if text == '©':
                continue
            score = item["SimilarityScore"]
            print("Score- ", score)
            meta = {"source": item["source"], "category": item["category"]}
            docs_and_scores.append(
                (Document(page_content=f"{text}", metadata=meta), score))
        return docs_and_scores

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
    ) -> List[Tuple[Document, float]]:
        embeddings = self._embedding.embed_query(query)
        docs_and_scores = self._similarity_search_with_score(embeddings=embeddings, k=k, user_query=query)
        return docs_and_scores

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        print("query", query)
        docs_and_scores = self.similarity_search_with_score(query, k=k)

        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        # Retrieves the docs with similarity scores
        docs = self._similarity_search_with_score(embeddings=embedding, k=fetch_k)

        # Re-ranks the docs using MMR
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )

        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        # compute the embeddings vector from the query string
        embeddings = self._embedding.embed_query(query)

        docs = self.max_marginal_relevance_search_by_vector(
            embeddings,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        return docs
