import argparse

from config import TOP_K_DEFAULT
from rag.generator import generate_answer
from rag.retriever import retrieve_tweets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    result = retrieve_tweets(
        query=args.query,
        k=args.top_k,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    retrieval_query = result["retrieval_query"]
    query_variants = result["query_variants"]
    query_variants_debug = result.get("query_variants_debug", [])
    docs = result["docs"]

    print(f"\nORIGINAL QUERY: {args.query}")
    print(f"RETRIEVAL QUERY: {retrieval_query}")
    print(f"QUERY VARIANTS: {query_variants}")
    if query_variants_debug:
        print(f"QUERY VARIANT DEBUG: {query_variants_debug}")
    print(f"DATE FROM: {args.date_from}")
    print(f"DATE TO: {args.date_to}")
    print(f"RERANK POOL SIZE: {result.get('rerank_pool_size', 0)}")
    print(f"SUPPRESSED DUPLICATES: {len(result.get('suppressed_duplicate_ids', []))}")
    print("=" * 80)

    if not docs:
        print("No documents found.")
        return

    print("\nRETRIEVED DOCUMENTS\n")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] DATE: {doc.metadata.get('date', '')}")
        print(f"URL: {doc.metadata.get('post_url', '')}")
        print(f"RERANK SCORE: {doc.metadata.get('rerank_score', 0.0)}")
        print(f"FUSED SCORE: {doc.metadata.get('fused_score', 0.0)}")
        print(f"DENSE SCORE: {doc.metadata.get('dense_score', 0.0)}")
        print(f"BM25 SCORE: {doc.metadata.get('bm25_score', 0.0)}")
        print(f"BGE SPARSE SCORE: {doc.metadata.get('bge_sparse_score', 0.0)}")
        print(f"TEXT: {doc.page_content}")
        print("-" * 80)

    if args.no_generate:
        return

    time_range = f"{args.date_from or 'N/A'} - {args.date_to or 'N/A'}"
    answer = generate_answer(args.query, docs, time_range=time_range)
    print("\nMODEL ANSWER\n")
    print(answer)
    print("=" * 80)


if __name__ == "__main__":
    main()
