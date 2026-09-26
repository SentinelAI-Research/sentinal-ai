from backend.provenance.graph import ProvenanceGraph

"""
We don't want the ML code itself to understand NetworkX internals.

Instead:
NetworkX ProvenanceGraph
        ↓
provenance_features.py
        ↓
ML feature values
"""

def extract_provenance_features(
    graph: ProvenanceGraph,
    object_id: str,
) -> dict[str, int]:
    return {
        "provenance_depth": graph.get_depth(object_id),
        "transformation_count": graph.get_transformation_depth(object_id),
    }