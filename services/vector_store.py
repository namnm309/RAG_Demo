"""
ChromaVectorStore — persistent dual collections (system_kb + hr_kb).
Embeddings are supplied by the app (Ollama); Chroma stores vectors + metadata.
"""

from typing import List, Literal, Optional

import chromadb

from models.schemas import CollectionName, DocumentChunk

COSINE_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        system_collection: str = "system_kb",
        hr_collection: str = "hr_kb",
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._system_name = system_collection
        self._hr_name = hr_collection
        self._system = self._get_or_create_collection(system_collection)
        self._hr = self._get_or_create_collection(hr_collection)

    def _get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata=COSINE_COLLECTION_METADATA,
        )

    def _collection_for(self, collection: CollectionName):
        return self._system if collection == "system" else self._hr

    @property
    def is_ready(self) -> bool:
        return self.count("system") > 0 or self.count("hr") > 0

    def count(self, collection: CollectionName) -> int:
        col = self._collection_for(collection)
        return col.count()

    def add_chunks(self, chunks: List[DocumentChunk], collection: CollectionName) -> None:
        if not chunks:
            return
        col = self._collection_for(collection)
        col.add(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[self._chunk_to_metadata(c) for c in chunks],
        )

    def search(
        self,
        query_embedding: List[float],
        collection: CollectionName,
        top_k: int = 5,
        min_score: float = 0.0,
        owner_id: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[DocumentChunk]:
        col = self._collection_for(collection)
        where = self._build_where(collection, owner_id, doc_type)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = col.query(**kwargs)
        return self._parse_query_result(result, collection, min_score)

    def delete_by_source_file(
        self,
        collection: CollectionName,
        source_file: str,
        owner_id: Optional[str] = None,
    ) -> int:
        col = self._collection_for(collection)
        clauses = [{"source_file": source_file}]
        if collection == "hr" and owner_id:
            clauses.append({"owner_id": owner_id})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        existing = col.get(where=where, include=[])
        ids = existing.get("ids") or []
        if ids:
            col.delete(ids=ids)
        return len(ids)

    def delete_scoped(self, collection: CollectionName, owner_id: Optional[str] = None) -> int:
        if collection == "system":
            before = self._system.count()
            self._client.delete_collection(self._system_name)
            self._system = self._get_or_create_collection(self._system_name)
            return before

        col = self._hr
        if owner_id:
            existing = col.get(where={"owner_id": owner_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                col.delete(ids=ids)
            return len(ids)
        before = col.count()
        self._client.delete_collection(self._hr_name)
        self._hr = self._get_or_create_collection(self._hr_name)
        return before

    def count_for_owner(self, owner_id: str) -> int:
        if not owner_id:
            return 0
        data = self._hr.get(where={"owner_id": owner_id}, include=[])
        return len(data.get("ids") or [])

    def indexed_files(self, collection: CollectionName, owner_id: Optional[str] = None) -> List[str]:
        col = self._collection_for(collection)
        where = {"owner_id": owner_id} if collection == "hr" and owner_id else None
        kwargs = {"include": ["metadatas"]}
        if where:
            kwargs["where"] = where
        data = col.get(**kwargs)
        metadatas = data.get("metadatas") or []
        return sorted({m.get("source_file", "") for m in metadatas if m.get("source_file")})

    @staticmethod
    def _chunk_to_metadata(chunk: DocumentChunk) -> dict:
        meta = {
            "source_file": chunk.source_file,
            "chunk_index": int(chunk.chunk_index),
            "doc_type": chunk.doc_type,
            "knowledge_base": chunk.knowledge_base,
        }
        if chunk.owner_id:
            meta["owner_id"] = chunk.owner_id
        return meta

    @staticmethod
    def _build_where(
        collection: CollectionName,
        owner_id: Optional[str],
        doc_type: Optional[str],
    ) -> Optional[dict]:
        clauses = []
        if collection == "hr" and owner_id:
            clauses.append({"owner_id": owner_id})
        if doc_type:
            clauses.append({"doc_type": doc_type})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _parse_query_result(
        self,
        result: dict,
        collection: CollectionName,
        min_score: float,
    ) -> List[DocumentChunk]:
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: List[DocumentChunk] = []
        kb: str = collection
        for i, doc_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            score = 1.0 - float(distance)
            if score < min_score:
                continue
            meta = metadatas[i] if i < len(metadatas) else {}
            chunks.append(
                DocumentChunk(
                    id=doc_id,
                    text=documents[i] if i < len(documents) else "",
                    source_file=meta.get("source_file", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    embedding=[],
                    knowledge_base=meta.get("knowledge_base", kb),
                    owner_id=meta.get("owner_id"),
                    doc_type=meta.get("doc_type", "general"),
                    score=score,
                )
            )
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
