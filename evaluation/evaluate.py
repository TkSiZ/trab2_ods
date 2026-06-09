"""
evaluate.py — Avaliação RAGAS do PokéRAG
"""

import sys
import os
import asyncio
import textwrap
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from agent import PokeAgent

# ============================================================
# 0) Configurações
# ============================================================
JUDGE_MODEL  = "qwen2.5:3b"
INTERP_MODEL = "qwen2.5:3b"
EMBED_MODEL  = "intfloat/multilingual-e5-large"

# ============================================================
# 1) Perguntas e ground truths
# ============================================================
PERGUNTAS_E_RESPOSTAS = [
    {
        "question": "Quais são os melhores Pokémon do tipo Água?",
        "ground_truth": (
            "Some of the strongest Water type Pokémon include Gyarados, which has high Attack and Special Defense. "
            "Vaporeon has excellent HP and Special Defense. "
            "Starmie is fast with high Special Attack. "
            "Lapras has great bulk with high HP and Special Defense. "
            "Blastoise is a well-rounded Water type with solid Defense and Special Attack."
        ),
    },
    {
        "question": "Me fala sobre a lore do Gengar",
        "ground_truth": (
            "Gengar is a Ghost/Poison type Pokémon. "
            "It is said to emerge from darkness to steal the lives of those who become lost in mountains. "
            "It apparently wishes for a traveling companion. Since it was once human itself, "
            "it tries to create one by taking the lives of other humans. "
            "It is the final evolution of Gastly, evolving from Haunter when traded."
        ),
    },
    {
        "question": "Qual o melhor counter para Charizard?",
        "ground_truth": (
            "Charizard is a Fire/Flying type Pokémon, making it weak to Rock, Electric, and Water moves. "
            "Rock type moves deal 4x damage due to both Fire and Flying type weaknesses. "
            "Pokémon like Golem and Rhydon with Rock or Ground moves are strong counters. "
            "Water types like Blastoise and Vaporeon also counter Charizard effectively."
        ),
    },
    {
        "question": "Compare Bulbasaur, Charmander e Squirtle",
        "ground_truth": (
            "Bulbasaur is a Grass/Poison type starter with balanced stats, base stat total of 318, "
            "and high Special Attack and Special Defense. "
            "Charmander is a Fire type starter with high Speed and Special Attack, base stat total of 309. "
            "Squirtle is a Water type starter with high Defense and Special Defense, base stat total of 314. "
            "All three are Generation 1 starters that evolve twice into Venusaur, Charizard, and Blastoise."
        ),
    },
    {
        "question": "Quais Pokémon têm a habilidade Levitate?",
        "ground_truth": (
            "Pokémon with the Levitate ability include Gastly, Haunter, Gengar, Koffing, Weezing, "
            "Misdreavus, Mismagius, Flygon, Latias, Latios, and Rotom. "
            "Levitate grants immunity to Ground type moves. "
            "It is a common ability among Ghost and Poison type Pokémon."
        ),
    },
    {
        "question": "Tem algum Pokémon do tipo Grass e Ghost?",
        "ground_truth": (
            "Yes, there are Grass/Ghost type Pokémon. "
            "Trevenant and Phantump are Ghost/Grass types introduced in Generation 6. "
            "Gourgeist is another Ghost/Grass type that evolves from Pumpkaboo. "
            "Decidueye, the final evolution of Rowlet, is also a Grass/Ghost type introduced in Generation 7."
            "DHELMISE, BRAMBLIN, BRAMBLEGHAST, POLTCHAGEIST, SINISTCHA"
        ),
    },
]
#
# # ============================================================
# # 2) Coleta respostas — tudo dentro de um único asyncio.run()
# # ============================================================
# async def coletar_respostas():
#     agente = PokeAgent()
#     questions, answers, contexts, ground_truths = [], [], [], []
#
#     for item in PERGUNTAS_E_RESPOSTAS:
#         pergunta = item["question"]
#         print(f"  ❓ {pergunta}")
#
#         try:
#             resposta, steps = await agente.query(pergunta)
#         except Exception as e:
#             print(f"  ⚠️  Erro ao processar pergunta: {e}")
#             resposta = f"Erro: {e}"
#             steps = []
#
#         # Garante que a resposta é string
#         if not isinstance(resposta, str):
#             resposta = str(resposta)
#
#         # Coleta chunks de todos os steps
#         chunks = []
#         for step in steps:
#             raw = step.get("result_full", "")
#             if raw:
#                 partes = [p.strip() for p in raw.split("---") if p.strip()]
#                 chunks.extend(partes)
#
#         if not chunks:
#             chunks = ["Nenhum contexto recuperado."]
#
#         questions.append(pergunta)
#         answers.append(resposta)
#         contexts.append(chunks)
#         ground_truths.append(item["ground_truth"])
#
#         print(f"  💬 {resposta[:120]}...\n")
#
#     return questions, answers, contexts, ground_truths
#
#
# print("🔍 Executando perguntas no pipeline RAG...\n")
# questions, answers, contexts, ground_truths = asyncio.run(coletar_respostas())


questions = [item["question"]for item in PERGUNTAS_E_RESPOSTAS]

answers = [
    "Com base nas informações encontradas, os melhores Pokémon do tipo Água são: Alomomola - com habilidades de cura e regeneração, é um dos melhores Pokémon do tipo Água. Pelipper - com sua capacidade de transportar pequenos Pokémon e ovos em seu bico, é uma escolha popular entre os treinadores. Quaquaval - com suas pernas poderosas e habilidades de dança, é um Pokémon versátil e difícil de ser derrotado. Esses três Pokémon são considerados alguns dos melhores do tipo Água devido às suas habilidades únicas e capacidades de combate. No entanto, é importante lembrar que a escolha do melhor Pokémon depende do estilo de jogo e das preferências pessoais do treinador.",
    "O Gengar é um Pokémon de tipo fantasma e venenoso, conhecido por sua capacidade de se esconder nas sombras e absorver a calor das pessoas ao redor. Ele é capaz de causar uma sensação de frio intenso, fazendo com que as vítimas tremam. Além disso, o Gengar pode se esconder nas sombras das pessoas e esperar por uma oportunidade para atacar e roubar a vida delas. Ele é um Pokémon muito temido pelas suas habilidades misteriosas e sinistras.",
    "Com base nas informações encontradas, o melhor counter para Charizard é Ceruledge, pois ele é um tipo de fogo e fantasma, o que significa que ele tem uma vantagem sobre o Charizard, que é apenas do tipo fogo. Ceruledge também tem uma alta ataque e velocidade, o que permite que ele ataque rapidamente e com força. Além disso, sua habilidade 'flash-fire' pode aumentar a probabilidade de acertar um golpe crítico contra o Charizard. Portanto, se você estiver enfrentando um Charizard em uma batalha, é recomendável usar um Ceruledge como seu counter.",
    "Com base nas informações encontradas na base de conhecimento, aqui está uma comparação entre Bulbasaur, Charmander e Squirtle: Bulbasaur É um Pokémon tipo grama/planta. Tem HP: 45, Ataque: 49, Defesa: 49, Sp. Atk: 65, Sp. Def: 65, Velocidade: 45. Evolui para Ivysaur e depois Venusaur. Charmander É um Pokémon tipo fogo. Tem HP: 39, Ataque: 52, Defesa: 43, Sp. Atk: 60, Sp. Def: 50, Velocidade: 65. Evolui para Charmeleon e depois Charizard Squirtle É um Pokémon tipo água. Tem HP: 44, Ataque: 48, Defesa: 65, Sp. Atk: 50, Sp. Def: 64, Velocidade: 43. Evolui para Wartortle e depois Blastoise. Em resumo, cada um dos três Pokémon tem suas próprias habilidades e características únicas. Bulbasaur é um Pokémon tipo grama/planta com alta defesa e ataque especial, Charmander é um Pokémon tipo fogo com alta velocidade e ataque especial, enquanto Squirtle é um Pokémon tipo água com alta defesa e baixa velocidade. Se você está procurando por um Pokémon para treinar, é importante considerar as habilidades e características de cada um dos três antes de tomar uma decisão.",
    "Os Pokémon que têm a habilidade Levitate são: Vibrava Latios Latias Claydol Baltoy Flygon Sigilyph (com a habilidade Wonder-Skin) Chimecho Mismagius Misdreavus Essas são as informações encontradas na base de dados sobre Pokémon com a habilidade Levitate.",
    "Sim, existem Pokémon do tipo Grass e Ghost. Alguns exemplos incluem: Brambleghast Bramblin Decidueye Dhelmise Gourgeist-average Phantump Poltchageist Pumpkaboo-average Sinistcha Trevenant Esses Pokémon têm habilidades e estatísticas únicas, e alguns deles evoluem para outros formas.",
]

contexts = [
    [
        "Alomomola is the Caring Pokémon. It is a water-type Pokémon. Abilities: healer, hydration, regenerator (hidden). Base stats — HP: 165, Attack: 75, Defense: 80, Sp. Atk: 40, Sp. Def: 45, Speed: 65. Evolution line: alomomola. Lore: The special membrane enveloping Alomomola has the ability to heal wounds. Floating in the open sea is how they live. When they find a wounded Pokémon, they embrace it and bring it to shore. It gently holds injured and weak Pokémon in its fins. Its special membrane heals their wounds. It uses its special mucus to close the wounds of injured Pokémon. The reason for this behavior remains unknown. They float upon the open sea. Many water Pokémon gather in the area around Alomomola.",
        "Maractus is the Cactus Pokémon. It is a grass-type Pokémon. Abilities: water-absorb, chlorophyll, storm-drain (hidden). Base stats — HP: 75, Attack: 86, Defense: 67, Sp. Atk: 106, Sp. Def: 67, Speed: 60. Evolution line: maractus. Lore: It uses an up-tempo song and dance to drive away the bird Pokémon that prey on its flower seeds. Arid regions are their habitat. They move rhythmically, making a sound similar to maracas. When it moves rhythmically, it makes a sound similar to maracas, making the surprised Pokémon flee. With noises that could be mistaken for the rattles of maracas, it creates an upbeat rhythm, startling bird Pokémon and making them fly off in a hurry. Once each year, this Pokémon scatters its seeds. They’re jam-packed with nutrients, making them a precious food source out in the desert.",
        "Quaquaval is the Dancer Pokémon. It is a water and fighting-type Pokémon. Abilities: torrent, moxie (hidden). Base stats — HP: 85, Attack: 120, Defense: 80, Sp. Atk: 85, Sp. Def: 75, Speed: 85. Evolution line: quaxly → quaxwell → quaquaval. Lore: A single kick from a Quaquaval can send a truck rolling. This Pokémon uses its powerful legs to perform striking dances from far-off lands. Dancing in ways that evoke far-away places, this Pokémon mesmerizes all that see it. Flourishes of its decorative water feathers slice into its foes.",
        "Quaxwell is the Practicing Pokémon. It is a water-type Pokémon. Abilities: torrent, moxie (hidden). Base stats — HP: 70, Attack: 85, Defense: 65, Sp. Atk: 65, Sp. Def: 60, Speed: 65. Evolution line: quaxly → quaxwell → quaquaval. Lore: These Pokémon constantly run through shallow waters to train their legs, then compete with each other to see which of them kicks most gracefully. The hardworking Quaxwell observes people and Pokémon from various regions and incorporates their movements into its own dance routines.",
        "Lombre is the Jolly Pokémon. It is a water and grass-type Pokémon. Abilities: swift-swim, rain-dish, own-tempo (hidden). Base stats — HP: 60, Attack: 50, Defense: 50, Sp. Atk: 60, Sp. Def: 70, Speed: 50. Evolution line: lotad → lombre → ludicolo. Lore: It lives at the water’s edge where it is sunny. It sleeps on a bed of water grass by day and becomes active at night. LOMBRE is nocturnal - it will get active after dusk. It is also a mischief-maker. When this POKéMON spots anglers, it tugs on their fishing lines from beneath the surface and enjoys their consternation. LOMBRE’s entire body is covered by a slippery, slimy film. It feels horribly unpleasant to be touched by this POKéMON’s hands. LOMBRE is often mistaken for a human child. In the evening, it takes great delight in popping out of rivers and startling people. It feeds on aquatic moss that grows on rocks in the riverbed. It has a mischievous spirit. If it spots an angler, it will tug on the fishing line to interfere.",
        "Pelipper is the Water Bird Pokémon. It is a water and flying-type Pokémon. Abilities: keen-eye, drizzle, rain-dish (hidden). Base stats — HP: 60, Attack: 50, Defense: 100, Sp. Atk: 95, Sp. Def: 70, Speed: 65. Evolution line: wingull → pelipper. Lore: It is a messenger of the skies, carrying small Pokémon and eggs to safety in its bill. PELIPPER is a flying transporter that carries small POKéMON and eggs inside its massive bill. This POKéMON builds its nest on steep cliffs facing the sea. PELIPPER searches for food while in flight by skimming the wave tops. This POKéMON dips its large bill in the sea to scoop up food, then swallows everything in one big gulp. It skims the tops of waves as it flies. When it spots prey, it uses its large beak to scoop up the victim with water. It protects its eggs in its beak. It is a flying transporter that carries small POKéMON in its beak. It bobs on the waves to rest its wings.",
        "Capsakid is the Spicy Pepper Pokémon. It is a grass-type Pokémon. Abilities: chlorophyll, insomnia, klutz (hidden). Base stats — HP: 50, Attack: 62, Defense: 40, Sp. Atk: 62, Sp. Def: 40, Speed: 50. Evolution line: capsakid → scovillain. Lore: The more sunlight this Pokémon bathes in, the more spicy chemicals are produced by its body, and thus the spicier its moves become. Traditional Paldean dishes can be extremely spicy because they include the shed front teeth of Capsakid among their ingredients.",
        "Masquerain is the Eyeball Pokémon. It is a bug and flying-type Pokémon. Abilities: intimidate, unnerve (hidden). Base stats — HP: 70, Attack: 60, Defense: 62, Sp. Atk: 100, Sp. Def: 82, Speed: 80. Evolution line: surskit → masquerain. Lore: MASQUERAIN intimidates enemies with the eyelike patterns on its antennas. This POKéMON flaps its four wings to freely fly in any direction - even sideways and backwards - as if it were a helicopter. MASQUERAIN’s antennas have eyelike patterns that usually give it an angry look. If the “eyes” are droopy and appear sad, it is said to be a sign that a heavy rainfall is on its way. It intimidates foes with the large eyelike patterns on its antennae. Because it can’t fly if its wings get wet, it shelters itself from rain under large trees and eaves. The antennae have distinctive patterns that look like eyes. When it rains, they grow heavy, making flight impossible. Its antennae have eye patterns on them. Its four wings enable it to hover and fly in any direction.",
        "Basculegion-male is the Big Fish Pokémon. It is a water and ghost-type Pokémon. Abilities: swift-swim, adaptability, mold-breaker (hidden). Base stats — HP: 120, Attack: 112, Defense: 65, Sp. Atk: 80, Sp. Def: 75, Speed: 78. Evolution line: basculin → basculegion. Lore: Clads itself in the souls of comrades that perished before fulfilling their goals of journeying upstream. No other species throughout all Hisui's rivers is Basculegion's equal.",
        "Carvanha is the Savage Pokémon. It is a water and dark-type Pokémon. Abilities: rough-skin, speed-boost (hidden). Base stats — HP: 45, Attack: 90, Defense: 20, Sp. Atk: 65, Sp. Def: 20, Speed: 65. Evolution line: carvanha → sharpedo. Lore: CARVANHA’s strongly developed jaws and its sharply pointed fangs pack the destructive power to rip out boat hulls. Many boats have been attacked and sunk by this POKéMON. If anything invades CARVANHA’s territory, it will swarm and tear at the intruder with its pointed fangs. On its own, however, this POKéMON turns suddenly timid. CARVANHA attack ships in swarms, making them sink. Although it is said to be a very vicious POKéMON, it timidly flees as soon as it finds itself alone. It lives in massive rivers that course through jungles. It swarms prey that enter its territory. They swarm any foe that invades their territory. Their sharp fangs can tear out boat hulls.)",
    ],
    [
        "Gengar is the Shadow Pokémon. It is a ghost and poison-type Pokémon. Abilities: cursed-body. Base stats — HP: 60, Attack: 65, Defense: 60, Sp. Atk: 130, Sp. Def: 75, Speed: 110. Evolution line: gastly → haunter → gengar. Lore: Under a full moon, this POKéMON likes to mimic the shadows of people and laugh at their fright. A GENGAR is close by if you feel a sudden chill. It may be trying to lay a curse on you. It steals heat from its surround­ ings. If you feel a sudden chill, it is certain that a GENGAR appeared. To steal the life of its target, it slips into the prey's shadow and silently waits for an opportunity. Hiding in people's shadows at night, it absorbs their heat. The chill it causes makes the victims shake.)"
    ],
    [
        "Charizard is the Flame Pokémon. It is a fire and flying-type Pokémon. Abilities: blaze, solar-power (hidden). Base stats — HP: 78, Attack: 84, Defense: 78, Sp. Atk: 109, Sp. Def: 85, Speed: 100. Evolution line: charmander → charmeleon → charizard. Lore: Spits fire that is hot enough to melt boulders. Known to cause forest fires unintentionally. When expelling a blast of super hot fire, the red flame at the tip of its tail burns more intensely. If CHARIZARD be­ comes furious, the flame at the tip of its tail flares up in a whitish- blue color. Breathing intense, hot flames, it can melt almost any­ thing. Its breath inflicts terrible pain on enemies. It uses its wings to fly high. The temperature of its fire increases as it gains exper­ ience in battle.",
        "Charmeleon is the Flame Pokémon. It is a fire-type Pokémon. Abilities: blaze, solar-power (hidden). Base stats — HP: 58, Attack: 64, Defense: 58, Sp. Atk: 80, Sp. Def: 65, Speed: 80. Evolution line: charmander → charmeleon → charizard. Lore: When it swings its burning tail, it elevates the temperature to unbearably high levels. Tough fights could excite this POKéMON. When excited, it may blow out bluish- white flames. It is very hot­ headed by nature, so it constantly seeks opponents. It calms down only when it wins. It has a barbaric nature. In battle, it whips its fiery tail around and slashes away with sharp claws. If it becomes agitated during battle, it spouts intense flames, incinerating its surroundings.",
        "Ceruledge is the Fire Blades Pokémon. It is a fire and ghost-type Pokémon. Abilities: flash-fire, weak-armor (hidden). Base stats — HP: 75, Attack: 125, Defense: 80, Sp. Atk: 60, Sp. Def: 100, Speed: 85. Evolution line: charcadet → armarouge → ceruledge. Lore: The fiery blades on its arms burn fiercely with the lingering resentment of a sword wielder who fell before accomplishing their goal. An old set of armor steeped in grudges caused this Pokémon’s evolution. Ceruledge cuts its enemies to pieces without mercy.)"
    ],
    [
        "Charmander is the Lizard Pokémon. It is a fire-type Pokémon. Abilities: blaze, solar-power (hidden). Base stats — HP: 39, Attack: 52, Defense: 43, Sp. Atk: 60, Sp. Def: 50, Speed: 65. Evolution line: charmander → charmeleon → charizard. Lore: Obviously prefers hot places. When it rains, steam is said to spout from the tip of its tail. The flame at the tip of its tail makes a sound as it burns. You can only hear it in quiet places. The flame on its tail shows the strength of its life force. If it is weak, the flame also burns weakly. The flame on its tail indicates CHARMANDER's life force. If it is healthy, the flame burns brightly. If it's healthy, the flame on the tip of its tail will burn vigor­ ously, even if it gets a bit wet.",
        "Charmeleon is the Flame Pokémon. It is a fire-type Pokémon. Abilities: blaze, solar-power (hidden). Base stats — HP: 58, Attack: 64, Defense: 58, Sp. Atk: 80, Sp. Def: 65, Speed: 80. Evolution line: charmander → charmeleon → charizard. Lore: When it swings its burning tail, it elevates the temperature to unbearably high levels. Tough fights could excite this POKéMON. When excited, it may blow out bluish- white flames. It is very hot­ headed by nature, so it constantly seeks opponents. It calms down only when it wins. It has a barbaric nature. In battle, it whips its fiery tail around and slashes away with sharp claws. If it becomes agitated during battle, it spouts intense flames, incinerating its surroundings.",
        "Squirtle is the Tiny Turtle Pokémon. It is a water-type Pokémon. Abilities: torrent, rain-dish (hidden). Base stats — HP: 44, Attack: 48, Defense: 65, Sp. Atk: 50, Sp. Def: 64, Speed: 43. Evolution line: squirtle → wartortle → blastoise. Lore: After birth, its back swells and hardens into a shell. Powerfully sprays foam from its mouth. Shoots water at prey while in the water. Withdraws into its shell when in danger. The shell is soft when it is born. It soon becomes so resilient, prod­ ding fingers will bounce off it. The shell, which hardens soon after it is born, is resilient. If you poke it, it will bounce back out. When it feels threatened, it draws its legs inside its shell and sprays water from its mouth.)",
    ],
    [
        "Vibrava is the Vibration Pokémon. It is a ground and dragon-type Pokémon. Abilities: levitate. Base stats — HP: 50, Attack: 70, Defense: 50, Sp. Atk: 50, Sp. Def: 50, Speed: 70. Evolution line: trapinch → vibrava → flygon. Lore: To make prey faint, VIBRAVA generates ultrasonic waves by vigorously making its two wings vibrate. This POKéMON’s ultrasonic waves are so powerful, they can bring on headaches in people. VIBRAVA’s wings have not yet completed the process of growing. Rather than flying long distances, they are more useful for generating ultrasonic waves by vibrating. It looses ultrasonic waves by rubbing its wings together. Since a VIBRAVA’s wings are still in the process of growing, it can only fly short distances. It generates ultrasonic waves by violently flapping its wings. After making its prey faint, it melts the prey with acid. It violently shudders its wings, generating ultrasonic waves to induce headaches in people.",
        "Latios is the Eon Pokémon. It is a dragon and psychic-type Pokémon. Abilities: levitate. Base stats — HP: 80, Attack: 90, Defense: 80, Sp. Atk: 130, Sp. Def: 110, Speed: 110. Evolution line: latios. Lore: LATIOS has the ability to make its foe see an image of what it has seen or imagines in its head. This POKéMON is intelligent and understands human speech. LATIOS will only open its heart to a TRAINER with a compassionate spirit. This POKéMON can fly faster than a jet plane by folding its forelegs to minimize air resistance. Even in hiding, it can detect the locations of others and sense their emotions since it has telepathy. Its intelligence allows it to understand human languages. It has a docile temperament and dislikes fighting. Tucking in its forelegs, it can fly faster than a jet plane. A highly intelligent Pokémon. By folding back its wings in flight, it can overtake jet planes.",
        "Latias is the Eon Pokémon. It is a dragon and psychic-type Pokémon. Abilities: levitate. Base stats — HP: 80, Attack: 80, Defense: 90, Sp. Atk: 110, Sp. Def: 130, Speed: 110. Evolution line: latias. Lore: LATIAS is highly sensitive to the emotions of people. If it senses any hostility, this POKéMON ruffles the feathers all over its body and cries shrilly to intimidate the foe. LATIAS is highly intelligent and capable of understanding human speech. It is covered with a glass-like down. The POKéMON enfolds its body with its down and refracts light to alter its appearance. They make a small herd of only several members. They rarely make contact with people or other POKéMON. They disappear if they sense enemies. It can telepathically communicate with people. It changes its appearance using its down that refracts light. Its body is covered with a down that can refract light in such a way that it becomes invisible.",
        "Claydol is the Clay Doll Pokémon. It is a ground and psychic-type Pokémon. Abilities: levitate. Base stats — HP: 60, Attack: 70, Defense: 105, Sp. Atk: 70, Sp. Def: 120, Speed: 75. Evolution line: baltoy → claydol. Lore: CLAYDOL are said to be dolls of mud made by primitive humans and brought to life by exposure to a mysterious ray. This POKéMON moves about while levitating. CLAYDOL is an enigma that appeared from a clay statue made by an ancient civilization dating back 20,000 years. This POKéMON shoots beams from both its hands. A CLAYDOL sleeps while hovering in midair. Its arms are separate from its body. They are kept floating by the POKéMON’s manipulation of psychic power. It appears to have been born from clay dolls made by ancient people. It uses telekinesis to float and move. An ancient clay figurine that came to life as a Pokémon from exposure to a mysterious ray of light.",
        "Baltoy is the Clay Doll Pokémon. It is a ground and psychic-type Pokémon. Abilities: levitate. Base stats — HP: 40, Attack: 40, Defense: 55, Sp. Atk: 40, Sp. Def: 70, Speed: 55. Evolution line: baltoy → claydol. Lore: BALTOY moves while spinning around on its one foot. Primitive wall paintings depicting this POKéMON living among people were discovered in some ancient ruins. As soon as it spots others of its kind, BALTOY congregates with them and then begins crying noisily in unison. This POKéMON sleeps while cleverly balancing itself on its one foot. A BALTOY moves by spinning on its single foot. It has been depicted in murals adorning the walls of a once-bustling city in an ancient age. It was discovered in ancient ruins. While moving, it constantly spins. It stands on one foot even when asleep. It moves by spinning on its foot. It is a rare Pokémon that was discovered in ancient ruins.",
        'Flygon is the Mystic Pokémon. It is a ground and dragon-type Pokémon. Abilities: levitate. Base stats — HP: 80, Attack: 100, Defense: 80, Sp. Atk: 80, Sp. Def: 80, Speed: 100. Evolution line: trapinch → vibrava → flygon. Lore: FLYGON is nicknamed “the elemental spirit of the desert.” Because its flapping wings whip up a cloud of sand, this POKéMON is always enveloped in a sandstorm while flying. FLYGON whips up a sandstorm by flapping its wings. The wings create a series of notes that sound like singing. Because the “singing” is the only thing that can be heard in a sandstorm, this POKéMON is said to be the desert spirit. The flapping of its wings sounds like singing. To prevent detection by enemies, it hides itself by flapping up a cloud of desert sand. It hides itself by kicking up desert sand with its wings. Red covers shield its eyes from sand. It whips up sandstorms with powerful flaps of its wings. It is known as “The Desert Spirit.”',
        "Sigilyph is the Avianoid Pokémon. It is a psychic and flying-type Pokémon. Abilities: wonder-skin, magic-guard, tinted-lens (hidden). Base stats — HP: 72, Attack: 58, Defense: 80, Sp. Atk: 103, Sp. Def: 80, Speed: 97. Evolution line: sigilyph. Lore: They never vary the route they fly, because their memories of guarding an ancient city remain steadfast. The guardians of an ancient city, they use their psychic power to attack enemies that invade their territory. The guardians of an ancient city, they always fly the same route while keeping watch for invaders. Psychic power allows these Pokémon to fly. Some say they were the guardians of an ancient city. Others say they were the guardians’ emissaries. A discovery was made in the desert where Sigilyph fly. The ruins of what may have been an ancient city were found beneath the sands.",
        "Chimecho is the Wind Chime Pokémon. It is a psychic-type Pokémon. Abilities: levitate. Base stats — HP: 75, Attack: 50, Defense: 80, Sp. Atk: 95, Sp. Def: 90, Speed: 65. Evolution line: chingling → chimecho. Lore: CHIMECHO makes its cries echo inside its hollow body. When this POKéMON becomes enraged, its cries result in ultrasonic waves that have the power to knock foes flying. In high winds, CHIMECHO cries as it hangs from a tree branch or the eaves of a building using a suction cup on its head. This POKéMON plucks berries with its long tail and eats them. They fly about very actively when the hot season arrives. They communicate among themselves using seven different and distinguishing cries. It travels by riding on winds. It cleverly uses its long tail to pluck nuts and berries, which it loves to eat. To knock foes flying, it makes the air shudder with its cries. It converses using seven cries.",
        "Mismagius is the Magical Pokémon. It is a ghost-type Pokémon. Abilities: levitate. Base stats — HP: 60, Attack: 60, Defense: 60, Sp. Atk: 105, Sp. Def: 105, Speed: 105. Evolution line: misdreavus → mismagius. Lore: Its cries sound like incantations. Those hearing it are tormented by headaches and hallucinations. It chants incantations. While they usually torment targets, some chants bring happiness. Its cry sounds like an incantation. It is said the cry may rarely be imbued with happiness-giving power. Its cries sound like incantations to torment the foe. It appears where you least expect it. It appears as if from nowhere—muttering incantations, placing curses, and giving people terrifying visions.",
        "Misdreavus is the Screech Pokémon. It is a ghost-type Pokémon. Abilities: levitate. Base stats — HP: 60, Attack: 60, Defense: 60, Sp. Atk: 85, Sp. Def: 85, Speed: 85. Evolution line: misdreavus → mismagius. Lore: It likes playing mischievous tricks such as screaming and wailing to startle people at night. It loves to bite and yank people's hair from behind without warning, just to see their shocked reactions. It loves to watch people it's scar­ ed. It frightens them by screaming loudly or appear­ ing suddenly. MISDREAVUS frightens people with a creepy, sobbing cry. The POKéMON apparently uses its red spheres to absorb the fearful feelings of foes and turn them into nutrition. A MISDREAVUS frightens people with a creepy, sobbing cry. It apparently uses its red spheres to absorb the fear of foes as its nutrition.)",
    ],
    [
        "Brambleghast [grass/ghost]: Brambleghast is the Tumbleweed Pokémon. It is a grass and ghost-type Pokémon. Abilities: wind-rider, infiltrator (hidden). Base stats — HP: 55, Attack: 115, Defense: 70, Sp. Atk: 80, Sp. Def: 70, Speed: 90. Evolution line: bramblin → brambleghast.",
        "Bramblin [grass/ghost]: Bramblin is the Tumbleweed Pokémon. It is a grass and ghost-type Pokémon. Abilities: wind-rider, infiltrator (hidden). Base stats — HP: 40, Attack: 65, Defense: 30, Sp. Atk: 45, Sp. Def: 35, Speed: 60. Evolution line: bramblin → brambleghast.",
        "Decidueye [grass/ghost]: Decidueye is the Arrow Quill Pokémon. It is a grass and ghost-type Pokémon. Abilities: overgrow, long-reach (hidden). Base stats — HP: 78, Attack: 107, Defense: 75, Sp. Atk: 100, Sp. Def: 100, Speed: 70. Evolution line: rowlet → dartrix → decidueye.",
        "Dhelmise [ghost/grass]: Dhelmise is the Sea Creeper Pokémon. It is a ghost and grass-type Pokémon. Abilities: steelworker. Base stats — HP: 70, Attack: 131, Defense: 100, Sp. Atk: 86, Sp. Def: 90, Speed: 40. Evolution line: dhelmise.",
        "Gourgeist-average [ghost/grass]: Gourgeist-average is the Pumpkin Pokémon. It is a ghost and grass-type Pokémon. Abilities: pickup, frisk, insomnia (hidden). Base stats — HP: 65, Attack: 90, Defense: 122, Sp. Atk: 58, Sp. Def: 75, Speed: 84. Evolution line: pumpkaboo → gourgeist.",
        "Phantump [ghost/grass]: Phantump is the Stump Pokémon. It is a ghost and grass-type Pokémon. Abilities: natural-cure, frisk, harvest (hidden). Base stats — HP: 43, Attack: 70, Defense: 48, Sp. Atk: 50, Sp. Def: 60, Speed: 38. Evolution line: phantump → trevenant.",
        "Poltchageist [grass/ghost]: Poltchageist is the Matcha Pokémon. It is a grass and ghost-type Pokémon. Abilities: hospitality, heatproof (hidden). Base stats — HP: 40, Attack: 45, Defense: 45, Sp. Atk: 74, Sp. Def: 54, Speed: 50. Evolution line: poltchageist → sinistcha",
        "Pumpkaboo-average [ghost/grass]: Pumpkaboo-average is the Pumpkin Pokémon. It is a ghost and grass-type Pokémon. Abilities: pickup, frisk, insomnia (hidden). Base stats — HP: 49, Attack: 66, Defense: 70, Sp. Atk: 44, Sp. Def: 55, Speed: 51. Evolution line: pumpkaboo → gourgeist",
        "Sinistcha [grass/ghost]: Sinistcha is the Matcha Pokémon. It is a grass and ghost-type Pokémon. Abilities: hospitality, heatproof (hidden). Base stats — HP: 71, Attack: 60, Defense: 106, Sp. Atk: 121, Sp. Def: 80, Speed: 70. Evolution line: poltchageist → sinistcha",
        "Trevenant [ghost/grass]: Trevenant is the Elder Tree Pokémon. It is a ghost and grass-type Pokémon. Abilities: natural-cure, frisk, harvest (hidden). Base stats — HP: 85, Attack: 110, Defense: 76, Sp. Atk: 65, Sp. Def: 82, Speed: 56. Evolution line: phantump → trevenant"
    ]
]

ground_truths = [item["ground_truth"]for item in PERGUNTAS_E_RESPOSTAS]

# ============================================================
# 3) Monta EvaluationDataset
# ============================================================
# Por:
from datasets import Dataset
ragas_dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

# ============================================================
# 4) Configura juiz e embeddings
# ============================================================
print("⚙️  Configurando RAGAS com qwen3:32b como juiz...\n")

run_config = RunConfig(
    timeout=180,
    max_retries=5,
    max_wait=30,
    max_workers=1,
)

judge_llm = LangchainLLMWrapper(
    ChatOllama(model=JUDGE_MODEL, temperature=0),
    run_config=run_config,
)
judge_embed = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBED_MODEL)
)

metricas = [
    Faithfulness(llm=judge_llm),
    AnswerRelevancy(llm=judge_llm, embeddings=judge_embed),
    ContextPrecision(llm=judge_llm),
    ContextRecall(llm=judge_llm),
]

# ============================================================
# 5) Avalia
# ============================================================
print("📊 Rodando avaliação RAGAS...\n")

resultado = evaluate(
    dataset=ragas_dataset,
    metrics=metricas,
    run_config=run_config,
    raise_exceptions=False,
)

df = resultado.to_pandas()

COLUNAS      = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
COL_PERGUNTA = "user_input" if "user_input" in df.columns else "question"

print("\n" + "="*65)
print("           RESULTADOS DA AVALIAÇÃO RAGAS")
print("="*65)
print(df[[COL_PERGUNTA] + COLUNAS].to_string(index=False))

print("\n📈 Médias:")
medias = {}
for c in COLUNAS:
    if c in df.columns:
        col_values = df[c].dropna()
        if len(col_values) > 0:
            v = col_values.mean()
            medias[c] = v
            barra = "█" * int(v * 20)
            print(f"  {c:<22}: {v:.4f}  {barra}  (n={len(col_values)}/{len(df)})")
        else:
            print(f"  {c:<22}: sem dados válidos")

os.makedirs("evaluation", exist_ok=True)
csv_path = "evaluation/ragas_resultados.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n💾 Resultados salvos em: {csv_path}")

# ============================================================
# 6) Intérprete LLM leve
# ============================================================
print("\n" + "="*65)
print("       INTERPRETAÇÃO AUTOMÁTICA (qwen2.5:3b)")
print("="*65 + "\n")

linhas = []
for _, row in df.iterrows():
    linha = f"- Pergunta: \"{row[COL_PERGUNTA]}\"\n"
    for c in COLUNAS:
        v = row.get(c, float("nan"))
        try:
            linha += f"  {c}={'NaN' if math.isnan(v) else f'{v:.2f}'} | "
        except (TypeError, ValueError):
            linha += f"  {c}=N/A | "
    linhas.append(linha.rstrip(" | "))

medias_str = " | ".join(f"{k}={v:.2f}" for k, v in medias.items()) if medias else "sem dados"

PROMPT = f"""Você é um especialista em avaliação de sistemas RAG.

Abaixo estão os resultados de uma avaliação RAGAS de um pipeline RAG sobre Pokémon.

MÉTRICAS (0 a 1, maior = melhor):
- faithfulness: resposta fiel ao contexto? (baixo = alucinação)
- answer_relevancy: resposta relevante à pergunta? (baixo = resposta tangencial)
- context_precision: chunks recuperados são precisos? (baixo = ruído no retriever)
- context_recall: contexto cobre a resposta esperada? (baixo = retriever perdendo info)

MÉDIAS: {medias_str}

DETALHAMENTO:
{chr(10).join(linhas)}

Com base nesses resultados:
1. Identifique os pontos críticos do pipeline.
2. Explique em linguagem simples o que cada problema significa na prática.
3. Sugira melhorias concretas e priorizadas.
4. Dê uma nota geral de 0 a 10 e justifique.

Responda em português, de forma clara e objetiva.
"""

print("🧠 Chamando qwen2.5:3b para interpretar...\n")
interp_llm = ChatOllama(model=INTERP_MODEL, temperature=0.3)

try:
    resp = interp_llm.invoke(PROMPT)
    interpretacao = resp.content

    for linha in interpretacao.splitlines():
        print(textwrap.fill(linha, width=80) if linha.strip() else "")

    interp_path = "evaluation/ragas_interpretacao.txt"
    with open(interp_path, "w", encoding="utf-8") as f:
        f.write("INTERPRETAÇÃO AUTOMÁTICA — qwen2.5:3b\n")
        f.write("="*65 + "\n\n")
        f.write(interpretacao)

    print(f"\n💾 Interpretação salva em: {interp_path}")

except Exception as e:
    print(f"⚠️  Erro ao chamar o intérprete: {e}")