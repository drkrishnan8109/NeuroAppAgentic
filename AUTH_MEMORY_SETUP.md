# Login and memory setup

1. In Google Cloud, create an OIDC web client. Register
   `http://localhost:8501/oauth2callback` locally and the exact deployed
   `https://<app>.streamlit.app/oauth2callback` URI for production.
2. Create a Supabase Free project and copy its Postgres connection string from
   **Project Settings → Database**. Use the session-pooler connection string for
   serverless deployments when available.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` locally,
   replace every placeholder, and put the same TOML in Community Cloud Secrets.
4. Keep `MEMORY_USER_PEPPER` permanently unchanged. Rotating it changes every
   derived user ID and makes existing memories unreachable without migration.
5. Install dependencies and start the app:

   ```bash
   python -m pip install -r requirements.txt
   streamlit run app/streamlit_app.py
   ```

The app creates the `neuro_user_preferences` pgvector collection through mem0.
Memory reads fail open, so a paused Supabase Free project does not prevent RAG
answers. Supabase Free projects may pause after inactivity; resume the project in
the dashboard when memory remains empty after a period of non-use.

Extraction policy is versioned in `src/memory_schema.py`. Change `CATEGORIES` and
`CUSTOM_INSTRUCTIONS` there and bump `SCHEMA_VERSION` whenever the taxonomy or
meaning changes. Existing memories are not automatically reclassified.

