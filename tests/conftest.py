"""
Pytest configuration.

Sets required env vars *before* any `app.*` module is imported. This matters
because several modules initialize module-level singletons at import time
using these settings:
  - app/services/llm_client.py configures the genai SDK
  - app/db/database.py creates the SQLAlchemy engine
  - app/rag/vector_store.py creates the Chroma PersistentClient

Because Python caches imports, these singletons are only created once per
test session - so env vars must be correct on the *first* import, and
per-test isolation of the DB/vector-store paths must happen here (once),
not inside a per-test fixture (too late).
"""
import os
import tempfile

os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-pytest")

_test_data_dir = tempfile.mkdtemp(prefix="ai_learning_assistant_tests_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_test_data_dir}/test.db")
os.environ.setdefault("CHROMA_PERSIST_DIR", f"{_test_data_dir}/chroma")
