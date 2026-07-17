"""
RAG tool backed by a small DUMMY knowledge base for Phase 3 development.

Deliberately uses the same embedding model (all-MiniLM-L6-v2) and the same
kind of persistent Chroma store as your real Document Q&A project, so
swapping this out later is a config change (CHROMA_PERSIST_DIR /
CHROMA_COLLECTION_NAME in config.py), not a rewrite. The retrieval logic
here is what would run against your real store too.

Run `python scripts/seed_dummy_kb.py` once before using this tool, to build
the collection from the placeholder docs below.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from orchestrator.config import config

# Placeholder documents — replace this list (or better, swap the whole
# collection for your real project's) whenever you're ready to go live.
# Kept intentionally generic/small so retrieval behavior is easy to eyeball.
DUMMY_DOCS = [
    {
        "id": "doc1",
        "text": (
            "The company's remote work policy allows employees to work from "
            "home up to 3 days per week. Requests for full remote work require "
            "manager approval and are reviewed quarterly."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Employees are entitled to 18 days of paid annual leave per year, "
            "plus 10 public holidays. Unused leave can be carried over up to "
            "5 days into the next calendar year."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "The expense reimbursement process requires submitting receipts "
            "within 30 days of purchase through the internal finance portal. "
            "Approvals typically take 3-5 business days."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "New hires go through a 2-week onboarding program covering company "
            "tools, security training, and a meet-and-greet with their team. "
            "A dedicated buddy is assigned for the first month."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "The internal engineering wiki documents the on-call rotation "
            "schedule, incident response runbooks, and escalation contacts "
            "for each service team."
        ),
    },
]


class RAGTool:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME
        )

    def seed(self):
        """Build the dummy collection from DUMMY_DOCS. Idempotent-ish — clears
        and re-adds each time, so it's safe to re-run while iterating."""
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        texts = [d["text"] for d in DUMMY_DOCS]
        ids = [d["id"] for d in DUMMY_DOCS]
        embeddings = self.embedder.encode(texts).tolist()

        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)
        print(f"Seeded {len(DUMMY_DOCS)} documents into '{config.CHROMA_COLLECTION_NAME}'")

    def query(self, question: str, top_k: int = 2) -> dict:
        """
        Retrieve the top_k most relevant chunks for a question.

        Returns: {"context": str, "sources": list[str]}
        - context: retrieved chunks joined together, ready to drop into a
          synthesis prompt.
        - sources: the doc IDs used, for citing back to the user.
        """
        query_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        documents = results["documents"][0] if results["documents"] else []
        ids = results["ids"][0] if results["ids"] else []

        context = "\n\n".join(documents)
        return {"context": context, "sources": ids}
