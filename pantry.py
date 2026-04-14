import logging
from typing import List, Dict

logger = logging.getLogger("NomNomAI.Pantry")

# Simple in-memory dict to store user ingredients.
# Keys are session_ids, values are lists of ingredient strings.
_pantry_store: Dict[str, List[str]] = {}

def get_pantry(session_id: str) -> List[str]:
    """Retrieve the pantry for a given session."""
    if not session_id:
        logger.warning("get_pantry called with empty session_id.")
        return []
    items = _pantry_store.get(session_id, [])
    logger.info(f"Fetched {len(items)} items for session: {session_id}")
    return items

def set_pantry(session_id: str, ingredients: List[str]) -> List[str]:
    """Overwrite the pantry for a given session."""
    if not session_id:
        logger.warning("set_pantry called with empty session_id. Ignoring.")
        return []
    _pantry_store[session_id] = ingredients
    logger.info(f"Saved {len(ingredients)} items for session: {session_id}")
    return _pantry_store[session_id]

def clear_pantry(session_id: str):
    """Clear out a user's pantry."""
    if session_id in _pantry_store:
        del _pantry_store[session_id]
        logger.info(f"Cleared pantry for session: {session_id}")
