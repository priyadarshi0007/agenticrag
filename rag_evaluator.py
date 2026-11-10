import json
from typing import List, Dict
from llama_index.core import Settings
from prompts import JUDGE_PROMPT_TEMPLATE

class RAGEvaluator:
    """Evaluate retrieval and generation quality of your RAG system."""

    def __init__(self):
        self.results = []

    def evaluate_retrieval(self, query: str, retrieved_sources: List[str], expected_sources: List[str] = None) -> Dict:
        """
        Evaluate if retrieval found relevant documents.

        Args:
            query: User's question
            retrieved_sources: List of PDF filenames returned by retrieval
            expected_sources: Optional ground truth documents that should be retrieved

        Returns:
            Dict with precision, recall metrics
        """
        if expected_sources is None:
            expected_sources = []

        retrieved_set = set(retrieved_sources)
        expected_set = set(expected_sources)

        # Precision: of retrieved docs, how many were relevant?
        precision = (len(retrieved_set & expected_set) / len(retrieved_set)) if retrieved_set else 0

        # Recall: of relevant docs, how many did we retrieve?
        recall = (len(retrieved_set & expected_set) / len(expected_set)) if expected_set else 1

        return {
            "query": query,
            "retrieved_sources": list(retrieved_sources),
            "expected_sources": list(expected_sources),
            "precision": precision,
            "recall": recall,
            "f1": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }

    def evaluate_context_sufficiency(self, query: str, context: str) -> Dict:
        """
        Use the judge LLM to assess if retrieved context is sufficient.

        Args:
            query: User's question
            context: Retrieved context from documents

        Returns:
            Dict with judge evaluation and confidence
        """
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, context=context[:2000])
        evaluation = str(Settings.llm.complete(judge_prompt)).strip().upper()

        is_sufficient = "PASS" in evaluation

        return {
            "query": query,
            "context_length": len(context),
            "judge_verdict": evaluation,
            "is_sufficient": is_sufficient
        }

    def evaluate_retrieval_quality(self, query: str, retrieved_context: str, retrieved_sources: List[str]) -> Dict:
        """
        Holistic evaluation: does retrieved context answer the query?

        Returns metrics on retrieval quality for learning purposes.
        """
        # 1. Context length
        context_len = len(retrieved_context)

        # 2. Judge's assessment
        sufficiency = self.evaluate_context_sufficiency(query, retrieved_context)

        # 3. Source diversity (did we retrieve from multiple docs?)
        unique_sources = len(set(retrieved_sources))

        result = {
            "query": query,
            "context_length": context_len,
            "judge_verdict": sufficiency["judge_verdict"],
            "is_sufficient": sufficiency["is_sufficient"],
            "num_sources": unique_sources,
            "sources": list(set(retrieved_sources))
        }

        self.results.append(result)
        return result

    def summarize_evaluation(self) -> Dict:
        """Show aggregate stats across all evaluated queries."""
        if not self.results:
            return {"error": "No evaluations yet"}

        total = len(self.results)
        sufficient = sum(1 for r in self.results if r["is_sufficient"])
        avg_context_len = sum(r["context_length"] for r in self.results) / total
        avg_sources = sum(r["num_sources"] for r in self.results) / total

        return {
            "total_queries": total,
            "sufficient_context_rate": sufficient / total,  # % of queries with good retrieval
            "avg_context_length": avg_context_len,
            "avg_sources_per_query": avg_sources,
            "results": self.results
        }

    def save_evaluation_report(self, filename: str = "evaluation_report.json"):
        """Save evaluation results for analysis."""
        summary = self.summarize_evaluation()
        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Evaluation report saved to {filename}")
