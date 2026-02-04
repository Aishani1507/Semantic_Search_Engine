
# Command-Line Interface (CLI) for the Search Engine
# - Accepts user query and retrieval parameters
# - Supports embedding-only and hybrid retrieval modes
# - Prints ranked search results to the console

import argparse
from retrieval.search import semantic_search, hybrid_search_rrf

def main():

    # Argument parser configuration
    parser = argparse.ArgumentParser(
        description="Semantic Search Engine (Dense + Hybrid Retrieval)"
    )

    # Search query provided by the user
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query string"
    )

    # Number of top-ranked documents to return
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)"
    )
    
    # Retrieval mode selection
    parser.add_argument(
        "--mode",
        choices=["embedding", "hybrid"],
        default="embedding",
        help="Retrieval mode: embedding or hybrid"
    )

    # Parse command-line arguments
    args = parser.parse_args()


    # Edge case handling
    if not args.query.strip():
        print(" Error: Empty query provided")
        return

    # Execute retrieval based on selected mode
    if args.mode == "embedding":
        results = semantic_search(args.query, args.top_k)
    else:
        results = hybrid_search_rrf(args.query, args.top_k)
    
    # Handle case where no results are returned
    if not results:
        print(" No results found")
        return

    # Display ranked search results

    for rank, res in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Document ID:", res["pid"])
        print("Score:", round(res["score"], 4))
        print("Snippet:", res["text"])

# Entry point for CLI execution
if __name__ == "__main__":
    main()
