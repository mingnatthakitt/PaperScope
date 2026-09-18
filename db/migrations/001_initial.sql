CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS papers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  sha256 TEXT NOT NULL UNIQUE,
  title TEXT,
  authors JSONB NOT NULL DEFAULT '[]'::jsonb,
  doi TEXT,
  arxiv_id TEXT,
  source_url TEXT,
  original_path TEXT NOT NULL,
  page_count INT,
  status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  stage TEXT NOT NULL DEFAULT 'queued',
  status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
  progress INT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
  message TEXT NOT NULL DEFAULT 'Queued for ingestion',
  attempts INT NOT NULL DEFAULT 0,
  locked_until TIMESTAMPTZ,
  error_code TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper_sections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  title TEXT,
  section_index INT NOT NULL,
  page_start INT,
  page_end INT,
  UNIQUE (paper_id, section_index)
);

CREATE TABLE IF NOT EXISTS paper_pages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  page_number INT NOT NULL,
  image_path TEXT NOT NULL,
  width INT,
  height INT,
  text_content TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (paper_id, page_number)
);

CREATE TABLE IF NOT EXISTS paper_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  section_id UUID REFERENCES paper_sections(id),
  chunk_index INT NOT NULL,
  content TEXT NOT NULL,
  page_start INT,
  page_end INT,
  section_title TEXT,
  chunk_type TEXT NOT NULL DEFAULT 'text',
  embedding vector(1024),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (paper_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS paper_visuals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  page_number INT NOT NULL,
  visual_type TEXT NOT NULL,
  figure_label TEXT,
  caption TEXT,
  visual_description TEXT NOT NULL,
  visual_interpretation TEXT,
  page_image_path TEXT NOT NULL,
  cropped_image_path TEXT,
  embedding vector(1024),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS paper_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (paper_id, model, prompt_version)
);

CREATE TABLE IF NOT EXISTS paper_graphs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  analysis_id UUID NOT NULL REFERENCES paper_analyses(id),
  diagram_type TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS paper_graph_nodes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  graph_id UUID NOT NULL REFERENCES paper_graphs(id) ON DELETE CASCADE,
  node_key TEXT NOT NULL,
  label TEXT NOT NULL,
  node_type TEXT NOT NULL,
  explanation TEXT NOT NULL,
  evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
  embedding vector(1024),
  UNIQUE (graph_id, node_key)
);

CREATE TABLE IF NOT EXISTS paper_graph_edges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  graph_id UUID NOT NULL REFERENCES paper_graphs(id) ON DELETE CASCADE,
  source_node_key TEXT NOT NULL,
  target_node_key TEXT NOT NULL,
  label TEXT,
  relation TEXT
);

CREATE INDEX IF NOT EXISTS papers_status_idx ON papers(status);
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs(status, locked_until);
CREATE INDEX IF NOT EXISTS paper_chunks_paper_id_idx ON paper_chunks(paper_id);
CREATE INDEX IF NOT EXISTS paper_visuals_paper_id_idx ON paper_visuals(paper_id);
CREATE INDEX IF NOT EXISTS paper_graph_nodes_graph_id_idx ON paper_graph_nodes(graph_id);

CREATE INDEX IF NOT EXISTS paper_chunks_embedding_hnsw
  ON paper_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS paper_visuals_embedding_hnsw
  ON paper_visuals USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS paper_graph_nodes_embedding_hnsw
  ON paper_graph_nodes USING hnsw (embedding vector_cosine_ops);
