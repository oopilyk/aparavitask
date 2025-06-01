from neo4j import GraphDatabase
import os

NEO4J_URI = "bolt://localhost:7687"  # Docker default
NEO4J_USER = "neo4j"
NEO4J_PASS = "strongpassword123"  # match your docker run password

# Global variable to store the Neo4j driver
neo4j_driver = None

def connect_to_neo4j(uri, user, password):
    """
    Establish a connection to Neo4j and return the driver.
    """
    global neo4j_driver
    print("Connecting to Neo4j...")
    try:
        neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 AS test")
            print("✅ Connected to Neo4j. Test result:", result.single()["test"])
        return neo4j_driver
    except Exception as e:
        print("❌ Failed to connect to Neo4j:", e)
        raise e

def close_connection():
    """
    Close the Neo4j connection.
    """
    global neo4j_driver
    if neo4j_driver:
        neo4j_driver.close()
        print("Closed Neo4j connection.")
        neo4j_driver = None

def insert_graph_data(entities, relationships):
    """
    Insert entities and relationships into the Neo4j graph database.
    """
    global neo4j_driver
    if not neo4j_driver:
        raise Exception("Neo4j driver is not initialized. Call connect_to_neo4j first.")
    
    with neo4j_driver.session() as session:
        for name in entities:
            print(f"🔄 Inserting entity: {name}")
            session.run("MERGE (:Entity {name: $name})", name=name)

        for rel in relationships:
            print(f"🔗 Creating relationship: {rel}")
            session.run("""
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[:RELATION {type: $relation, extra: $extra}]->(b)
            """, source=rel["source"], target=rel["target"],
                 relation=rel["relation"], extra=str(rel.get("extra", {})))

# Example usage:
if __name__ == "__main__":
    try:
        connect_to_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
        entities = ["Alice", "Bob", "Charlie"]
        relationships = [
            {"source": "Alice", "target": "Bob", "relation": "KNOWS", "extra": {"since": 2020}},
            {"source": "Bob", "target": "Charlie", "relation": "WORKS_WITH"}
        ]
        insert_graph_data(entities, relationships)
    finally:
        close_connection()