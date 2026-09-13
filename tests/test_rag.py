import shutil
import tempfile
import unittest
from pathlib import Path

from rag.ingestion import DOCS_DIR, chunk_document, ingest_documents, load_documents
from rag.rag_service import (
    NO_POLICY_RETRIEVED_MESSAGE,
    format_rag_context_for_prompt,
    rebuild_index,
    retrieve_business_context,
)
from rag.retriever import PolicyRetriever
from rag.vectorstore import FAISSVectorStore


class TestRAGKnowledgeLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the default index is built and loaded
        rebuild_index()

    def test_document_ingestion(self):
        """Verify reading documents from rag/documents/ and metadata chunking."""
        docs = load_documents(DOCS_DIR)
        self.assertGreaterEqual(len(docs), 5, "Expected at least 5 policy documents")

        doc_names = {d["filename"] for d in docs}
        expected_docs = {
            "inventory_policy.md",
            "payment_policy.md",
            "supplier_policy.md",
            "expense_policy.md",
            "business_rules.md",
        }
        self.assertTrue(
            expected_docs.issubset(doc_names),
            f"Missing expected documents. Found: {doc_names}",
        )

        sample_doc = docs[0]
        chunks = chunk_document(
            doc_text=sample_doc["text"],
            filename=sample_doc["filename"],
            document_name=sample_doc["document_name"],
        )
        self.assertGreater(len(chunks), 0, "Chunking should produce at least one chunk")

        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("source", chunk)
            self.assertIn("document_name", chunk)
            self.assertIn("chunk_id", chunk)
            self.assertEqual(chunk["source"], sample_doc["filename"])
            self.assertTrue(chunk["chunk_id"].startswith(sample_doc["filename"]))

    def test_vector_index_creation(self):
        """Verify FAISS vector index creation, dimension, and disk persistence."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            store = FAISSVectorStore()
            test_chunks = [
                {
                    "content": "Acme Retail requires 1.5 months of safety stock for all SKU categories.",
                    "source": "inventory_policy.md",
                    "document_name": "inventory_policy",
                    "chunk_id": "inventory_policy.md_chunk_0",
                },
                {
                    "content": "Standard customer payment terms are Net 30 days.",
                    "source": "payment_policy.md",
                    "document_name": "payment_policy",
                    "chunk_id": "payment_policy.md_chunk_0",
                },
            ]
            store.add_chunks(test_chunks)

            self.assertEqual(store.index.ntotal, 2)
            self.assertEqual(store.index.d, 384)

            # Test save and load
            store.save(temp_dir)
            loaded_store = FAISSVectorStore()
            success = loaded_store.load(temp_dir)
            self.assertTrue(success)
            self.assertEqual(loaded_store.index.ntotal, 2)
            self.assertEqual(len(loaded_store.chunks), 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_retrieval(self):
        """Verify that relevant queries retrieve matching policy chunks."""
        queries_and_expected_sources = [
            ("What is the inventory safety stock and coverage policy?", "inventory_policy.md"),
            ("What are customer payment terms and overdue escalation rules?", "payment_policy.md"),
            ("What is the supplier payment policy and early discount terms?", "supplier_policy.md"),
            ("What are the discretionary spending limits and OpEx rules?", "expense_policy.md"),
            ("What is the minimum gross margin benchmark and cash runway rule?", "business_rules.md"),
        ]

        for query, expected_source in queries_and_expected_sources:
            results = retrieve_business_context(query, top_k=3)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            sources = [r["source"] for r in results]
            self.assertIn(
                expected_source,
                sources,
                f"Query '{query}' failed to retrieve expected source '{expected_source}'. Got: {sources}",
            )

    def test_metadata_preservation(self):
        """Verify structured output maintains content, source, and chunk_id metadata."""
        results = retrieve_business_context("supplier payment delay terms", top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for item in results:
            self.assertIn("content", item)
            self.assertIn("source", item)
            self.assertIn("chunk_id", item)
            self.assertTrue(len(item["content"]) > 0)
            self.assertTrue(item["source"].endswith(".md"))
            self.assertIn(item["source"], item["chunk_id"])

    def test_unrelated_query_handling(self):
        """Verify similarity/relevance filtering rejects unrelated queries."""
        unrelated_queries = [
            "How to bake a sourdough chocolate cake?",
            "What is the orbital speed of the International Space Station?",
            "French grammar rules for irregular verbs",
        ]

        for query in unrelated_queries:
            results = retrieve_business_context(query, top_k=3)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["content"], NO_POLICY_RETRIEVED_MESSAGE)
            self.assertEqual(results[0]["source"], "none")
            self.assertEqual(results[0]["chunk_id"], "none")

            prompt_text = format_rag_context_for_prompt(results)
            self.assertEqual(prompt_text, NO_POLICY_RETRIEVED_MESSAGE)

    def test_empty_knowledge_base_handling(self):
        """Verify graceful handling when vector store / index is completely empty."""
        empty_store = FAISSVectorStore()
        retriever = PolicyRetriever(vectorstore=empty_store)

        results = retriever.retrieve("What is safety stock?", top_k=5)
        self.assertEqual(results, [])

        formatted = format_rag_context_for_prompt(results)
        self.assertEqual(formatted, NO_POLICY_RETRIEVED_MESSAGE)

    def test_analysis_agent_rag_prompt_structure(self):
        """Verify Analysis Agent prompt contains the 4 required sections and policy context."""
        from agents.analysis_agent import AnalysisAgent
        from backend.digital_twin.simulation import Scenario

        agent = AnalysisAgent()
        scenario = Scenario(sales_growth=0.3, payment_delay_days=15)
        sim_result = {
            "baseline": {"revenue": 100000, "current_cash": 50000},
            "projected": {"revenue": 130000, "ending_cash": 40000, "inventory_risk": "Low"},
            "risk": {"overall": "Medium", "inventory": "Low"},
        }

        prompt = agent.build_prompt(scenario, sim_result)

        # Check required sections in exact specified order
        pos_facts = prompt.find("AUTHORITATIVE SIMULATOR FACTS")
        pos_scenario = prompt.find("SCENARIO")
        pos_rag = prompt.find("RETRIEVED BUSINESS KNOWLEDGE")
        pos_inst = prompt.find("ANALYSIS INSTRUCTIONS")

        self.assertNotEqual(pos_facts, -1)
        self.assertNotEqual(pos_scenario, -1)
        self.assertNotEqual(pos_rag, -1)
        self.assertNotEqual(pos_inst, -1)
        self.assertTrue(pos_facts < pos_scenario < pos_rag < pos_inst, "Prompt sections out of order")

        # Check that customer payment policy was retrieved for payment_delay_days scenario
        self.assertIn("payment_policy.md", prompt)

        # Verify custom rag_context injection works
        custom_rag = [
            {
                "content": "Custom policy for testing.",
                "source": "custom_policy.md",
                "chunk_id": "custom_chunk_1",
            }
        ]
        custom_prompt = agent.build_prompt(scenario, sim_result, rag_context=custom_rag)
        self.assertIn("custom_policy.md", custom_prompt)
        self.assertIn("Custom policy for testing.", custom_prompt)

    def test_deterministic_repeated_retrieval(self):
        """Verify that identical queries produce identical results in identical order."""
        query = "What is the safety stock and inventory coverage policy?"
        res1 = retrieve_business_context(query, top_k=3)
        res2 = retrieve_business_context(query, top_k=3)

        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1["source"], r2["source"])
            self.assertEqual(r1["chunk_id"], r2["chunk_id"])
            self.assertEqual(r1["content"], r2["content"])


if __name__ == "__main__":
    unittest.main()


