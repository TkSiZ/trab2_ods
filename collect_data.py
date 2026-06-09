"""
PokéRAG — Coleta de dados via PokéAPI
Baixa dados de todos os Pokémon e salva como JSON na pasta data/.

Uso: python scripts/collect_data.py
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL   = "https://pokeapi.co/api/v2"
OUTPUT_DIR = Path("data")
DELAY      = 0.3   # segundos entre requests
LIMIT      = 1025   # 151 = Gen 1 | 251 = Gen 1-2 | 1025 = todos

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PokéRAG/1.0 (projeto acadêmico)"})

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_json(url: str) -> dict:
    r = SESSION.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def build_pokemon_doc(name: str) -> dict | None:
    """Busca dados de um Pokémon e retorna um documento estruturado."""
    try:
        poke    = get_json(f"{BASE_URL}/pokemon/{name}")
        species = get_json(poke["species"]["url"])
    except Exception as e:
        print(f"  Erro em {name}: {e}")
        return None

    # --- Flavor text em inglês (lore) ---
    flavor_entries = [
        e["flavor_text"].replace("\n", " ").replace("\f", " ")
        for e in species.get("flavor_text_entries", [])
        if e["language"]["name"] == "en"
    ]
    # remove duplicatas preservando ordem
    seen, flavor_texts = set(), []
    for t in flavor_entries:
        if t not in seen:
            seen.add(t)
            flavor_texts.append(t)

    # --- Descrição do genus (ex: "Seed Pokémon") ---
    genus = next(
        (g["genus"] for g in species.get("genera", []) if g["language"]["name"] == "en"),
        ""
    )

    # --- Tipos ---
    types = [t["type"]["name"] for t in poke["types"]]

    # --- Habilidades ---
    abilities = [
        {"name": a["ability"]["name"], "hidden": a["is_hidden"]}
        for a in poke["abilities"]
    ]

    # --- Movimentos (apenas nomes) ---
    moves = [m["move"]["name"] for m in poke["moves"]]

    # --- Stats base ---
    stats = {s["stat"]["name"]: s["base_stat"] for s in poke["stats"]}

    # --- Cadeia evolutiva ---
    evo_chain_url = species.get("evolution_chain", {}).get("url")
    evolution_chain = []
    if evo_chain_url:
        try:
            evo_data = get_json(evo_chain_url)
            def extract_chain(node):
                chain = [node["species"]["name"]]
                for evo in node.get("evolves_to", []):
                    chain.extend(extract_chain(evo))
                return chain
            evolution_chain = extract_chain(evo_data["chain"])
        except Exception:
            pass

    # --- Texto corrido para embedding ---
    lore_text = " ".join(flavor_texts[:5])  # até 5 entradas de lore
    abilities_text = ", ".join(
        f"{a['name']} (hidden)" if a["hidden"] else a["name"]
        for a in abilities
    )
    text = (
        f"{poke['name'].capitalize()} is the {genus}. "
        f"It is a {' and '.join(types)}-type Pokémon. "
        f"Abilities: {abilities_text}. "
        f"Base stats — HP: {stats.get('hp')}, Attack: {stats.get('attack')}, "
        f"Defense: {stats.get('defense')}, Sp. Atk: {stats.get('special-attack')}, "
        f"Sp. Def: {stats.get('special-defense')}, Speed: {stats.get('speed')}. "
        f"Evolution line: {' → '.join(evolution_chain)}. "
        f"Lore: {lore_text}"
    )

    return {
        "id":             poke["id"],
        "name":           poke["name"],
        "genus":          genus,
        "types":          types,
        "abilities":      abilities,
        "moves":          moves,
        "stats":          stats,
        "evolution_chain": evolution_chain,
        "flavor_texts":   flavor_texts,
        "text":           text,   # campo principal para embedding
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # lista de nomes
    print("Buscando lista de Pokémon...")
    list_data = get_json(f"{BASE_URL}/pokemon?limit={LIMIT}&offset=0")
    names = [p["name"] for p in list_data["results"]]
    print(f"Total: {len(names)} Pokémon\n")

    saved, skipped = 0, 0

    with tqdm(names, unit=" pokémon", colour="yellow") as pbar:
        for name in pbar:
            out_file = OUTPUT_DIR / f"{name}.json"

            if out_file.exists():
                saved += 1
                pbar.set_postfix(saved=saved, skipped=skipped, status="cached")
                continue

            doc = build_pokemon_doc(name)
            if doc is None:
                skipped += 1
                pbar.set_postfix(saved=saved, skipped=skipped, status="erro")
                continue

            out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            saved += 1
            pbar.set_postfix(saved=saved, skipped=skipped, status=name)
            time.sleep(DELAY)

    print(f"\n✓ Concluído — salvos: {saved} | erros: {skipped}")
    print(f"Dados em: {OUTPUT_DIR.resolve()}")