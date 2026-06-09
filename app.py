"""
PokéRAG — Interface Streamlit
Assistente inteligente de Pokémon com Agentic RAG.
"""

import asyncio
import nest_asyncio
nest_asyncio.apply()
import streamlit as st
import markdown as md
from agent import get_agent
from PIL import Image

icon = Image.open("assets/pokeball.png")

# ── Página ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PokéRAG",
    page_icon=icon,
    layout="centered",
)

# ── Estilos ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Inter:wght@400;500;600&display=swap');

  /* fundo com silhuetas de pokémon */
  .stApp {
    background-color: #0a0e1a;
    background-image:
      radial-gradient(ellipse at 10% 20%, rgba(42,117,187,0.08) 0%, transparent 50%),
      radial-gradient(ellipse at 90% 80%, rgba(255,203,5,0.06) 0%, transparent 50%),
      url("https://www.transparenttextures.com/patterns/dark-mosaic.png");
    color: #e8e8e8;
  }

  /* pokébola decorativa no fundo */
  .stApp::before {
    content: '';
    position: fixed;
    top: -120px;
    right: -120px;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle at 50% 50%,
      rgba(255,50,50,0.04) 0%,
      rgba(255,50,50,0.04) 45%,
      rgba(255,255,255,0.03) 45%,
      rgba(255,255,255,0.03) 55%,
      rgba(20,20,40,0.04) 55%,
      rgba(20,20,40,0.04) 100%
    );
    border: 2px solid rgba(255,255,255,0.03);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
  }

  /* título */
  .poke-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 1.5rem;
    color: #FFCB05;
    text-shadow: 3px 3px 0px #2A75BB, 6px 6px 0px rgba(0,0,0,0.3);
    text-align: center;
    padding: 1rem 0 0.25rem;
    letter-spacing: 2px;
  }
  .poke-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #6b7280;
    text-align: center;
    margin-bottom: 1.5rem;
    letter-spacing: 1px;
  }

  /* balão usuário */
  .msg-user {
    background: linear-gradient(135deg, #1e3a5f, #162d4a);
    border-left: 3px solid #2A75BB;
    border-radius: 0 12px 12px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  .msg-user .icon {
    display: inline-block;
    width: 44px;
    height: 44px;
    margin-right: 8px;
    vertical-align: middle;
    /* pokébola SVG inline */
    background: url("https://ssb.wiki.gallery/images/thumb/2/28/Pok%C3%A9mon_Trainer_%28solo%29_SSBU.png/250px-Pok%C3%A9mon_Trainer_%28solo%29_SSBU.png") no-repeat center/contain;
  }

  /* balão bot */
  .msg-bot {
    background: linear-gradient(135deg, #13132a, #0f0f1f);
    border-left: 3px solid #FFCB05;
    border-radius: 0 12px 12px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    line-height: 1.6;
  }
  .msg-bot .bot-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.5rem;
    font-family: 'Press Start 2P', monospace;
    font-size: 0.55rem;
    color: #FFCB05;
    letter-spacing: 1px;
  }
  .msg-bot .pokeball-icon {
    display: inline-block;
    width: 44px;
    height: 44px;
    background: url("https://cdn.iconscout.com/icon/free/png-512/free-pokemon-icon-svg-download-png-32211.png?f=webp&w=256") no-repeat center/contain;
  }
  /* markdown dentro do bot */
  .msg-bot p  { margin: 0.3rem 0; }
  .msg-bot ul { padding-left: 1.2rem; margin: 0.3rem 0; }
  .msg-bot ol { padding-left: 1.2rem; margin: 0.3rem 0; }
  .msg-bot li { margin: 0.2rem 0; }
  .msg-bot strong { color: #FFCB05; }
  .msg-bot h1, .msg-bot h2, .msg-bot h3 {
    color: #FFCB05;
    font-family: 'Press Start 2P', monospace;
    font-size: 0.7rem;
    margin: 0.6rem 0 0.3rem;
    letter-spacing: 1px;
  }
  .msg-bot code {
    background: #1f2937;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    font-size: 0.82rem;
    color: #6ee7b7;
  }

  /* steps do agente */
  .agent-step {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.45rem 0.75rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: #6ee7b7;
    margin: 0.2rem 0;
  }
  .tool-name  { color: #FFCB05; font-weight: 600; }
  .tool-input { color: #93c5fd; }

  /* input */
  .stChatInput {
  padding: 0.5rem 0 !important;
  }
  .stChatInput > div {
   background: linear-gradient(135deg, #13132a, #0f0f1f) !important;
   border: 2px solid #FFCB05 !important;
   border-radius: 30px !important;
   box-shadow: 0 0 12px rgba(255,203,5,0.15), 0 0 2px rgba(255,203,5,0.3) !important;
 }
 .stChatInput > div:focus-within {
   border-color: #FFCB05 !important;
   box-shadow: 0 0 20px rgba(255,203,5,0.25) !important;
 }
 .stChatInput textarea {
   color: #e8e8e8 !important;
   font-family: 'Inter', sans-serif !important;
   font-size: 0.9rem !important;
 }
 .stChatInput textarea::placeholder {
   color: #4b5563 !important;
   font-style: italic;
 }

  /* botão limpar */
  div[data-testid="stButton"] button {
    background: transparent;
    border: 1px solid #374151;
    color: #6b7280;
    font-size: 0.75rem;
    border-radius: 6px;
  }
  div[data-testid="stButton"] button:hover {
    border-color: #ef4444;
    color: #ef4444;
  }

  hr { border-color: #1f2937; margin: 0.8rem 0; }

  /* expander */
  .streamlit-expanderHeader {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
  }
     /* remove white top bar */
    header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    }
    # header[data-testid="stHeader"] * {
        display: none
    # }

    /* fix chat input background */
    .stChatInput > div {
        background: transparent !important;
    }
    .stChatInput textarea {
        border: none;
        box-shadow: none !important;
        color: #e8e8e8 !important;
    }

    .stChatInput > div:focus-within {
        border: 2px solid #FFCB05 !important;
        box-shadow: 0 0 20px rgba(255,203,5,0.25) !important;
    }
            
    .stChatInput textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    .stChatInput textarea {
        outline: none !important;
    }

    /* fix bottom bar behind chat input */
    .stBottom {
        background: transparent !important;
    }
    .stBottom > div {
        background: transparent !important;
    }
            
    div[data-baseweb="textarea"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 0.5rem 0 0;">
  <div style="font-size:2.5rem; margin-bottom:0.3rem;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Pok%C3%A9_Ball_icon.svg"
     width="48" height="48" style="vertical-align:middle; margin-right:12px;">
    <span class="poke-title">PokéRAG</span>
  </div>
  <div class="poke-subtitle">ASSISTENTE INTELIGENTE · AGENTIC RAG · LLAMA 3 + CHROMADB</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Estado ─────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Carregando Pokédex..."):
        st.session_state.agent = get_agent()

# ── Helper: converte markdown para HTML ────────────────────────────────────────

def to_html(text: str) -> str:
    return md.markdown(text, extensions=["nl2br"])

# ── Histórico ──────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">'
            f'<span class="icon"></span>{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        content_html = to_html(msg["content"])
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="bot-header"><span class="pokeball-icon"></span>POKÉRAG</div>'
            f'{content_html}'
            f'</div>',
            unsafe_allow_html=True
        )
        if msg.get("steps"):
            with st.expander("🔍 Passos do agente", expanded=False):
                for step in msg["steps"]:
                    st.markdown(
                        f'<div class="agent-step">'
                        f'[{step["iteration"]}] '
                        f'<span class="tool-name">{step["tool"]}</span>'
                        f'(<span class="tool-input">{step["input"]}</span>)'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ── Input ──────────────────────────────────────────────────────────────────────

exemplos = [
    "Quais são os melhores Pokémon do tipo Água?",
    "Me fala sobre a lore do Gengar",
    "Qual o melhor counter para Charizard?",
    "Compare Bulbasaur, Charmander e Squirtle",
    "Quais Pokémon têm a habilidade Levitate?",
    "Tem algum Pokémon do tipo Grass e Ghost?",
]

with st.expander("💡 Exemplos de perguntas", expanded=False):
    for ex in exemplos:
        if st.button(ex, key=ex):
            st.session_state._pending_input = ex
            st.rerun()

pergunta = st.chat_input("Consulte a Pokédex...")

if hasattr(st.session_state, "_pending_input"):
    pergunta = st.session_state._pending_input
    del st.session_state._pending_input

# ── Resposta ───────────────────────────────────────────────────────────────────

if pergunta:
    st.session_state.messages.append({"role": "user", "content": pergunta})
    st.markdown(
        f'<div class="msg-user">'
        f'<span class="icon"></span>{pergunta}'
        f'</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Consultando a Pokédex..."):
        resposta, steps = asyncio.run(
            st.session_state.agent.query(pergunta)
        )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": resposta,
        "steps":   steps,
    })

    content_html = to_html(resposta)
    st.markdown(
        f'<div class="msg-bot">'
        f'<div class="bot-header"><span class="pokeball-icon"></span>POKÉRAG</div>'
        f'{content_html}'
        f'</div>',
        unsafe_allow_html=True
    )

    if steps:
        with st.expander("🔍 Passos do agente", expanded=True):
            for step in steps:
                st.markdown(
                    f'<div class="agent-step">'
                    f'[{step["iteration"]}] '
                    f'<span class="tool-name">{step["tool"]}</span>'
                    f'(<span class="tool-input">{step["input"]}</span>)'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ── Rodapé ─────────────────────────────────────────────────────────────────────

st.markdown("---")
col1, col2 = st.columns([4, 1])
with col1:
    st.caption("PokéRAG · LLaMA 3.1 8B · nomic-embed-text · ChromaDB · LangChain")
with col2:
    if st.button("🗑️ Limpar"):
        st.session_state.messages = []
        st.rerun()