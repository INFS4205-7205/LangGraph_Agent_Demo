from typing import TypedDict, List
import csv
import os
import chromadb
from chromadb.config import Settings
from PIL import Image
from sentence_transformers import SentenceTransformer
import uuid

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# =========================
# Agent State
# =========================

class AgentState(TypedDict):
    messages: List[BaseMessage]


# =========================
# Dataset
# =========================

def load_courses(path: str = "courses.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

COURSES = load_courses()


# =========================
# Tools
# =========================

@tool
def search_courses(query: str) -> str:
    """Search the course dataset for relevant entries."""
    query_tokens = query.lower().split()
    scored = []

    for course in COURSES:
        text = " ".join(course.values()).lower()
        score = sum(token in text for token in query_tokens)
        if score > 0:
            scored.append((score, course))

    if not scored:
        return "No relevant courses found in the dataset."

    scored.sort(key=lambda x: x[0], reverse=True)

    return "\n".join(
        f"{c['code']}: {c['title']} ({c['level']}) - {c['description']}"
        for _, c in scored
    )


@tool
def calc(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        return "Error: invalid characters"

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def write_text(path: str, content: str) -> str:
    """Write text content to a local file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved {len(content)} characters to {path}"
    except Exception as e:
        return f"Error: {e}"

# =========================
# Image Indexing (CLIP + ChromaDB)
# =========================

class ImageIndex:
    def __init__(self, image_dir: str = "images"):
        self.image_dir = image_dir
        
        # Initialize ChromaDB (persistent)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="local_images")
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.model = SentenceTransformer('clip-ViT-B-32')
        
        self._index_images()

    def _index_images(self):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            
        # Get list of images on disk
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [
            f for f in os.listdir(self.image_dir) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        if not image_files:
            print(f"No images found in {self.image_dir}")
            return

        # Check what's already indexed to avoid re-embedding
        existing_ids = set(self.collection.get()["ids"])
        
        new_images = []
        new_ids = []
        new_metadatas = []
        
        print("Checking for new images to index...")
        for f in image_files:
            file_id = f"img_{f}" # Simple ID scheme
            if file_id not in existing_ids:
                try:
                    # Open and validate image
                    img_path = os.path.join(self.image_dir, f)
                    image = Image.open(img_path)
                    new_images.append(image)
                    new_ids.append(file_id)
                    new_metadatas.append({"filename": f})
                except Exception as e:
                    print(f"Skipping {f}: {e}")

        if new_images:
            print(f"Embedding and indexing {len(new_images)} new images...")
            embeddings = self.model.encode(new_images, normalize_embeddings=True)
            self.collection.add(
                embeddings=embeddings.tolist(),
                ids=new_ids,
                metadatas=new_metadatas
            )
            print("Indexing complete.")
        else:
            print("Index is up to date.")

    def search(self, query: str, k: int = 3):
        # Embed query
        query_emb = self.model.encode([query], normalize_embeddings=True)
        
        # Query Chroma
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k
        )
        
        # Parse results
        parsed = []
        if results and results['metadatas']:
            metas = results['metadatas'][0]
            dists = results['distances'][0]  # Chroma returns distances by default
            
            for i, meta in enumerate(metas):
                parsed.append((meta['filename'], dists[i]))
                
        return parsed

# Initialize global index
image_index = ImageIndex()

@tool
def search_images(query: str) -> str:
    """Search for images/memes/photos locally using a description."""
    results = image_index.search(query)
    if not results:
        return "No relevant images found."
    
    return "\n".join(
        f"Top K = 3 Images Found: {filename} (distance: {dist:.2f})" 
        for filename, dist in results
    )

TOOLS = [search_courses, calc, write_text, search_images]


# =========================
# LLM
# =========================

llm = ChatOllama(
    model="qwen3:0.6b-q4_K_M",
    temperature=0,
).bind_tools(TOOLS)


# =========================
# System Prompt
# =========================

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a careful assistant with access to tools.\n\n"
        "Guidelines:\n"
        "- Use calc for ANY arithmetic, even if it seems simple.\n"
        "- Use write_text when the user asks to write text content to a local file.\n"
        "- Use search_courses when the user asks about courses, degrees, or programs.\n"
        "- When search_courses returns results, summarize or filter them to answer the user's question.\n"
        "- Do NOT ask follow-up questions unless the request is ambiguous.\n"
        "- If you answer without tools and the answer could be wrong, that is a failure.\n"
        "- Prefer tools when accuracy matters."
    )
)

# =========================
# Graph Nodes
# =========================

def agent_node(state: AgentState):
    messages = state["messages"]

    # Find the last human message (original question)
    last_user_msg = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    if messages and messages[-1].type == "tool":
        # Tool just ran → force final answer grounded in results
        messages = messages + [
            SystemMessage(
                content=(
                    "You have received tool results.\n"
                    f"The original user question was:\n"
                    f"\"{last_user_msg}\"\n\n"
                    "Answer that question directly using the tool results.\n"
                    "Do NOT ask clarifying questions.\n"
                    "Do NOT call any more tools."
                )
            )
        ]
    else:
        messages = messages + [
            SystemMessage(
                content="Before answering, decide whether a tool would improve accuracy."
            )
        ]

    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}


tool_node = ToolNode(TOOLS)


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def print_tool_usage(messages):
    used_any = False

    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            used_any = True
            for call in m.tool_calls:
                print(f"[TOOL CALL] {call['name']}({call['args']})")

        if m.type == "tool":
            print(f"[TOOL RESULT] {m.content}")

    if not used_any:
        print("[TOOLS] No tools were used.")


# =========================
# Build LangGraph
# =========================

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        END: END,
    },
)

graph.add_edge("tools", "agent")

app = graph.compile()


# =========================
# Run loop
# =========================

if __name__ == "__main__":
    print("\nMinimal LangGraph Agent ready. Type 'exit' to quit.")

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        result = app.invoke(
            {
                "messages": [
                    SYSTEM_PROMPT,
                    HumanMessage(content=user),
                ]
            }
        )

        print(f"\nAgent: {result['messages'][-1].content}")
