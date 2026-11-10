JUDGE_PROMPT_TEMPLATE = """
You are a strict technical evaluator. Examine the retrieved context below and determine if it contains enough information to answer the user's question.

User Question: {query}
Retrieved Context: {context}

Respond with ONLY one word: either 'PASS' if the context is sufficient, or 'FAIL' if it is missing key details.
"""

REWRITER_PROMPT_TEMPLATE = """
You are a search query optimizer. Rewrite this technical query into clean keywords for vector search.

Example 1:
Original: How does Apache Iceberg handle atomic transactions?
Rewritten: Apache Iceberg atomic transactions ACID consistency metadata

Original: {query}
Rewritten (Output ONLY the raw keyword query string. No quotes, no markdown, no explanation):
"""

SYNTHESIS_PROMPT_TEMPLATE = """
You are an air-gapped technical assistant. You MUST answer the question using ONLY the provided context below. 
CRITICAL RULE: If the answer cannot be found explicitly within the provided context, state: "I cannot find this information in the provided technical documents." Do NOT use outside knowledge.

Context: 
{context}

Question: {query}
Answer:
"""