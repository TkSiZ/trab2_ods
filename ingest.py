"""
PokéRAG — Ingestão na base vetorial
Lê os JSONs da pasta data/ e indexa no ChromaDB com embeddings Ollama.

Uso: python scripts/ingest.py
"""

import json
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("data")
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "pokerag"
EMBED_MODEL     = "nomic-embed-text:latest"

# ── Carregar documentos ────────────────────────────────────────────────────────

def load_documents() -> list[Document]:
    docs = []
    files = sorted(DATA_DIR.glob("*.json"))
    skipped = 0

    print(f"Carregando {len(files)} arquivos de {DATA_DIR}/\n")

    for f in tqdm(files, unit=" arquivo", colour="cyan"):
        raw = f.read_text(encoding="utf-8").strip()

        # pula/remove arquivos vazios ou corrompidos
        if not raw:
            f.unlink()
            skipped += 1
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            f.unlink()
            skipped += 1
            continue

        docs.append(Document(
            page_content=data["text"],
            metadata={
                "name":            data["name"],
                "id":              data["id"],
                "types":           ",".join(data["types"]),
                "abilities":       ",".join(a["name"] for a in data["abilities"]),
                "moves":           ",".join(data["moves"][:20]),
                "evolution_chain": "->".join(data["evolution_chain"]),
                "source":          f"pokeapi/{data['name']}",
            }
        ))

    if skipped:
        print(f"\n  {skipped} arquivo(s) inválido(s) removido(s).")

    return docs


# ── Indexar no ChromaDB ────────────────────────────────────────────────────────

def ingest(docs: list[Document]):
    print(f"\nGerando embeddings com {EMBED_MODEL}...")
    print("(isso pode levar alguns minutos na primeira execução)\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )

    print(f"\n✓ {len(docs)} documentos indexados em '{VECTORSTORE_DIR}/'")
    return vectorstore


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_documents()
    ingest(docs)
    print("\nIngestão concluída. Rode o app com: streamlit run app.py")