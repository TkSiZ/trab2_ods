# ⚡ PokéRAG — Assistente Inteligente de Pokémon

Sistema de Agentic RAG especializado no domínio Pokémon.
O agente decide quais ferramentas usar, reformula queries e itera até gerar uma resposta fundamentada na base de conhecimento.

---

## Problema e domínio

Jogadores de Pokémon — especialmente no contexto competitivo (VGC/Smogon) — precisam consultar rapidamente informações sobre tipos, habilidades, lore, movimentos e cadeias evolutivas. As fontes existentes (Bulbapedia, Serebii) são ricas mas exigem navegação manual e não respondem perguntas em linguagem natural.

**PokéRAG** resolve isso com um agente conversacional que consulta uma base de conhecimento local e responde diretamente.

**Público-alvo:** jogadores competitivos e fãs da franquia que querem respostas rápidas sem navegar em múltiplas wikis.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                   Interface (Streamlit)              │
└──────────────────────┬──────────────────────────────┘
                       │ pergunta
┌──────────────────────▼──────────────────────────────┐
│                  Agente (LLaMA 3.1 8B)               │
│                                                      │
│  decide qual ferramenta usar e itera até ter         │
│  informação suficiente para responder                │
│                                                      │
│  ┌──────────────┐ ┌─────────────┐ ┌───────────────┐ │
│  │semantic_search│ │search_by_type│ │search_by_name │ │
│  └──────┬───────┘ └──────┬──────┘ └───────┬───────┘ │
└─────────┼────────────────┼────────────────┼─────────┘
          └────────────────▼────────────────┘
┌─────────────────────────────────────────────────────┐
│              ChromaDB (vetorstore local)             │
│         embeddings: nomic-embed-text via Ollama      │
│         ~1025 Pokémon indexados                      │
└─────────────────────────────────────────────────────┘
```

### Ferramentas do agente

| Ferramenta | Quando o agente usa |
|---|---|
| `semantic_search` | Perguntas em linguagem natural, lore, estratégias, comparações |
| `search_by_type` | Perguntas sobre um tipo específico (ex: "melhores do tipo Fogo") |
| `search_by_name` | Nome exato de um Pokémon mencionado pelo usuário |

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| LLM | LLaMA 3.1 8B via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vetorstore | ChromaDB (persistente local) |
| Framework | LangChain |
| Interface | Streamlit |
| Fonte dos dados | PokéAPI (https://pokeapi.co) |

---

## Instalação

### Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado e rodando

### 1. Modelos Ollama

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Dependências Python

```bash
pip install -r requirements.txt
```

---

## Execução

### Passo 1 — Coletar dados da PokéAPI

```bash
python scripts/collect_data.py
```

Baixa dados de todos os Pokémon e salva em `data/`. Para limitar a Gen 1 (mais rápido), edite `LIMIT = 151` em `collect_data.py`.

### Passo 2 — Indexar na base vetorial

```bash
python scripts/ingest.py
```

Gera embeddings e popula o ChromaDB em `vectorstore/`. Só precisa rodar uma vez.

### Passo 3 — Rodar o app

```bash
streamlit run app.py
```

Acesse em `http://localhost:8501`.

---

## Exemplos de uso

**Pergunta sobre lore:**
> "Me fala sobre a origem do Gengar"

**Comparação:**
> "Compare Bulbasaur, Charmander e Squirtle"

**Estratégia competitiva:**
> "Qual o melhor counter para Charizard?"

**Por tipo:**
> "Quais Pokémon do tipo Psíquico têm maior Sp. Atk?"

**Cadeia evolutiva:**
> "Como o Eevee evolui e quais são suas evoluções?"

---

## Decisões técnicas

**Por que LLaMA 3.1 8B?**
Bom equilíbrio entre qualidade de resposta e custo computacional local. Suporte sólido a português e raciocínio em múltiplos passos. Cabe em 12GB VRAM com quantização Q4.

**Por que nomic-embed-text?**
Modelo de embeddings open-source com bom desempenho em inglês (idioma da PokéAPI), leve (768 dims) e disponível diretamente via Ollama.

**Por que ChromaDB?**
Simples de configurar, persistência local sem servidor externo, suporte nativo a filtros por metadados (essencial para `search_by_type` e `search_by_name`).

**Por que 3 ferramentas separadas?**
Cada ferramenta tem força diferente: semântica cobre nuance, por tipo cobre queries categoriais, por nome cobre lookup exato. O agente aprende a combinar — ex: primeiro `search_by_name` para pegar o Pokémon, depois `semantic_search` para estratégia.

---

## Limitações conhecidas

- Base em inglês (PokéAPI retorna dados em EN); respostas em PT são geradas pelo LLM
- Dados de movesets e estratégias competitivas detalhadas não estão na base (só movenames)
- LLaMA 3.1 8B pode alucinar nomes de moves ou stats se não encontrar na base
- `search_by_type` usa filtro por metadados, então requer tipo em inglês exato

---

## Avaliação

Consulte `evaluation/` para métricas de recuperação e exemplos de perguntas avaliadas manualmente.

---

## Estrutura do projeto

```
pokerag/
├── app.py                  # Interface Streamlit
├── agent.py                # Agente Agentic RAG + ferramentas
├── requirements.txt
├── README.md
├── scripts/
│   ├── collect_data.py     # Coleta da PokéAPI
│   └── ingest.py           # Ingestão no ChromaDB
├── data/                   # JSONs dos Pokémon (gerado)
└── vectorstore/            # ChromaDB persistente (gerado)
```