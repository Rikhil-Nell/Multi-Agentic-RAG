-- Ensure the pgvector extension is enabled
create extension if not exists vector;

-- Create Table
create table public.documents (
  id bigint primary key generated always as identity,
  filename text not null,
  chunk_number integer not null,
  content text not null,
  embedding vector(1536) null,
  title text null,
  summary text null,
  created_at timestamp with time zone not null default timezone ('utc'::text, now()),
  constraint documents_filename_chunk_number_key unique (filename, chunk_number)
) TABLESPACE pg_default;

create index documents_embedding_idx on public.documents using ivfflat (embedding vector_cosine_ops) TABLESPACE pg_default;
-- Create Return Type
create type match_result as (
  filename text,
  content text,
  title text,
  summary text,
  similarity float
);

-- Create Function
create or replace function match_documents(
  query_embedding vector,
  match_count integer
)
returns setof match_result
language plpgsql
as $$
begin
  return query
  select
    d.filename,
    d.content,
    d.title,
    d.summary,
    1 - (d.embedding <=> query_embedding) as similarity
  from documents d
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;
