"""
PokéRAG — Agente Agentic RAG
Núcleo inteligente com três ferramentas: busca semântica, busca por tipo e busca por nome.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    collection_name: str   = "pokerag"
    vectorstore_dir: str   = "vectorstore"
    embed_model:     str   = "nomic-embed-text:latest"
    chat_model:      str   = "llama3.1:8b"
    temperature:     float = 0.1
    top_k:           int   = 4
    max_iterations:  int   = 6

# ── Agente ─────────────────────────────────────────────────────────────────────

class PokeAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        self.embeddings = OllamaEmbeddings(model=self.config.embed_model)
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            persist_directory=self.config.vectorstore_dir,
            embedding_function=self.embeddings,
        )
        self.llm = ChatOllama(
            model=self.config.chat_model,
            temperature=self.config.temperature,
        )

        # ── Ferramentas ──────────────────────────────────────────────────────

        @tool
        def semantic_search(query: str, top_k: int = 4) -> str:
            """
            Busca semântica na base de conhecimento Pokémon.
            Use para perguntas sobre lore, descrições, habilidades, movimentos,
            estratégias, comparações ou qualquer pergunta em linguagem natural.
            Pode ser chamada múltiplas vezes com queries diferentes.
            Use top_k maior (6-10) para comparações ou listas.
            Use top_k menor (1-3) para perguntas sobre um Pokémon específico.
            """
            top_k = max(1, min(top_k, 10))
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            if not results:
                return f"Nenhum resultado encontrado para: '{query}'"

            parts = []
            for i, (doc, score) in enumerate(results, 1):
                name = doc.metadata.get("name", "?").capitalize()
                parts.append(
                    f"[{i}] {name} (relevância: {score:.2f})\n{doc.page_content}"
                )
            return "\n\n---\n\n".join(parts)

        @tool
        def search_by_type(types: list) -> str:
            """
            Busca Pokémon por um ou mais tipos.
            Sempre use nomes em inglês: water, fire, grass, ghost, psychic, dragon, etc.
            Para um tipo: ["water"]
            Para múltiplos tipos (Pokémon que tenha TODOS): ["grass", "ghost"]
            Retorna apenas Pokémon que possuem TODOS os tipos informados simultaneamente.
            """
            types = [t.lower().strip() for t in types]

            all_results = self.vectorstore.get(include=["documents", "metadatas"])

            parts = []
            for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
                pokemon_types = [t.strip() for t in meta.get("types", "").split(",")]
                if all(t in pokemon_types for t in types):
                    name = meta.get("name", "?").capitalize()
                    type_str = "/".join(pokemon_types)
                    parts.append(f"- {name} [{type_str}]: {doc[:250]}...")

            if not parts:
                return f"Nenhum Pokémon com os tipos {' e '.join(types)} encontrado na base."

            label = " e ".join(types)
            return (
                f"Pokémon com os tipos {label} ({len(parts)} encontrados):\n\n"
                + "\n\n".join(parts[:10])
            )

        @tool
        def search_by_name(pokemon_name: str) -> str:
            """
            Busca dados completos de um Pokémon específico pelo nome exato.
            Use quando o usuário mencionar um Pokémon pelo nome.
            Retorna lore, tipos, habilidades, stats e cadeia evolutiva.
            """
            pokemon_name = pokemon_name.lower().strip()
            results = self.vectorstore.get(
                where={"name": pokemon_name},
                include=["documents", "metadatas"],
            )
            if not results["documents"]:
                fallback = self.vectorstore.similarity_search(pokemon_name, k=1)
                if fallback:
                    doc = fallback[0]
                    return (
                        f"Não encontrei '{pokemon_name}' exato. "
                        f"Resultado mais próximo:\n\n{doc.page_content}"
                    )
                return f"Pokémon '{pokemon_name}' não encontrado na base."

            doc  = results["documents"][0]
            meta = results["metadatas"][0]
            evo  = meta.get("evolution_chain", "").replace("->", " → ")

            return (
                f"{meta['name'].capitalize()}\n"
                f"Tipos: {meta.get('types','').replace(',','/')}\n"
                f"Habilidades: {meta.get('abilities','')}\n"
                f"Evolução: {evo}\n\n"
                f"{doc}"
            )

        self.tools     = [semantic_search, search_by_type, search_by_name]
        self.agent_llm = self.llm.bind_tools(self.tools)
        self._tool_map = {t.name: t for t in self.tools}

    # ── Loop de raciocínio ─────────────────────────────────────────────────────

    async def query(self, question: str) -> tuple[str, list[dict]]:
        """Processa uma pergunta com o agente. Retorna (resposta, log_steps)."""
        system_prompt = (
            "Você é PokéRAG, um assistente especialista em Pokémon. "
            "Você tem acesso a uma base de conhecimento com dados de todos os Pokémon. "
            "Use as ferramentas disponíveis para buscar informações antes de responder. "
            "Regras:\n"
            "- Sempre use ao menos uma ferramenta antes de responder.\n"
            "- Se a primeira busca não for suficiente, refine a query e busque novamente.\n"
            "- Cite os Pokémon e dados encontrados na base.\n"
            "- Se não encontrar informação suficiente, diga claramente.\n"
            "- Responda em português do Brasil de forma clara e objetiva.\n"
            "- Para search_by_type, SEMPRE use nomes de tipo em inglês "
            "(água=water, fogo=fire, grama=grass, fantasma=ghost, psíquico=psychic, etc).\n"
            "- Para perguntas com múltiplos tipos, passe todos na lista: search_by_type(['grass','ghost']).\n"
            "- Para perguntas sobre estratégia competitiva, use seu conhecimento + a base."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        steps = []

        for iteration in range(self.config.max_iterations):
            response = await self.agent_llm.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content, steps

            for tc in response.tool_calls:
                tool_name  = tc["name"]
                tool_args  = tc["args"]
                tool_input = str(list(tool_args.values())[0]) if tool_args else ""

                steps.append({
                    "iteration": iteration + 1,
                    "tool":      tool_name,
                    "input":     tool_input,
                })

                t      = self._tool_map.get(tool_name)
                result = t.invoke(tool_args) if t else f"Ferramenta '{tool_name}' não encontrada."
                result = result or "Ferramenta não retornou resultado."

                steps[-1]["result_preview"] = result[:200].replace("\n", " ")

                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        last = next(
            (m.content for m in reversed(messages) if hasattr(m, "content") and m.content),
            "Limite de iterações atingido sem resposta conclusiva."
        )
        return last, steps


# ── Singleton para o Streamlit ─────────────────────────────────────────────────

_agent_instance: Optional[PokeAgent] = None

def get_agent() -> PokeAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PokeAgent()
    return _agent_instance