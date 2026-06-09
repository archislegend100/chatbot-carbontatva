# Configuration

The CarbonTatva Industrial Energy Efficiency Copilot is configured entirely through environment variables. 
You can use a `.env` file in the root directory for local development.

## Required Variables

- `MISTRAL_API_KEY`: Your Mistral API key (required for LLM generation).

## Optional / Configurable Variables

- `MISTRAL_MODEL`: The specific Mistral model to use. Defaults to `mistral-small-latest`.
- `EMBEDDING_MODEL`: The embedding model for dense retrieval. Defaults to `BAAI/bge-large-en-v1.5`.
- `RERANKER_MODEL`: The reranker model. Defaults to `BAAI/bge-reranker-base`.
- `ENABLE_COLBERT`: Set to `true` to enable optional ColBERT retrieval for hard queries (requires `colbert-ai` installed). Defaults to `false`.
- `ENABLE_HYDE`: Set to `true` to enable Hypothetical Document Embeddings. Defaults to `false`.
- `ENABLE_MULTI_QUERY`: Set to `true` to enable multi-query routing and variants. Defaults to `true`.
- `ENABLE_VERIFICATION`: Set to `true` to enable an LLM verification pass before returning the final answer. Defaults to `true`.

Do **not** commit your `.env` file containing actual keys to version control.
