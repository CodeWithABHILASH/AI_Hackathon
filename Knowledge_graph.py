from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

# -------------------
# Setup Neo4j
# -------------------
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
neo_graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# -------------------
# Setup LLM
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini")
transformer = LLMGraphTransformer(llm=llm)

# -------------------
# LangGraph Nodes
# -------------------

def generate_kg(state):
    """
    Extract KG triples from raw text
    """
    text = state["text"]
    kg_doc = transformer.convert_to_graph(text)
    state["kg_doc"] = kg_doc
    return state

def ingest_neo4j(state):
    """
    Push KG into Neo4j
    """
    neo_graph.add_graph_document(state["kg_doc"])
    return state

def query_neo4j(state):
    """
    Example: Ask simple NL query
    """
    query = state.get("query")
    if not query:
        return state
    # convert query to Cypher (LLM or predefined mapping)
    cypher = f"MATCH (n) RETURN n LIMIT 5"  # simple example
    with driver.session() as session:
        result = session.run(cypher)
        state["result"] = [r.data() for r in result]
    return state

# -------------------
# Build LangGraph
# -------------------
graph = StateGraph()
graph.add_node("kg_gen", generate_kg)
graph.add_node("neo_ingest", ingest_neo4j)
graph.add_node("query_node", query_neo4j)

graph.add_edge(START, "kg_gen")
graph.add_edge("kg_gen", "neo_ingest")
graph.add_edge("neo_ingest", "query_node")
graph.add_edge("query_node", END)

workflow = graph.compile()

# -------------------
# Run Example
# -------------------
input_text = """
Arjun is an employee who knows Python and works on the Billing System project.
Sneha is skilled in React and works on the UI Team.
"""

response = workflow.invoke({"text": input_text, "query": "List all employees"})
print(response.get("result"))
