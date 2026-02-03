"""Named Entity Recognition for menu items and quantities."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import re
import json
from pathlib import Path


class EntityType(str, Enum):
    """Types of entities that can be extracted."""

    MENU_ITEM = "menu_item"
    QUANTITY = "quantity"
    MODIFIER = "modifier"
    DIETARY = "dietary"
    TEMPERATURE = "temperature"
    SIZE = "size"
    PERSON = "person"


@dataclass
class Entity:
    """Extracted entity."""

    type: EntityType
    value: str
    original_text: str
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "value": self.value,
            "original_text": self.original_text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction."""

    entities: List[Entity]
    text: str
    normalized_text: str = ""

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "text": self.text,
            "entity_count": len(self.entities),
            "entity_types": list(set(e.type.value for e in self.entities)),
        }

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get entities of a specific type."""
        return [e for e in self.entities if e.type == entity_type]


class EntityExtractor:
    """
    Extracts named entities from restaurant-related text.

    Supports:
    - Menu items (loaded from menu.json)
    - Quantities (numbers, words like 'two')
    - Modifiers (rare, medium, well-done, etc.)
    - Dietary restrictions (vegetarian, vegan, gluten-free)
    """

    # Number words to digits
    NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "a": 1,
        "an": 1,
    }

    # Common modifiers
    MODIFIERS = {
        "rare",
        "medium rare",
        "medium",
        "medium well",
        "well done",
        "extra",
        "no",
        "without",
        "with",
        "on the side",
        "light",
        "extra",
        "double",
    }

    # Temperature modifiers
    TEMPERATURE = {"hot", "cold", "warm", "iced", "room temperature"}

    # Size modifiers
    SIZE = {"small", "medium", "large", "regular", "extra large", "half", "full"}

    # Dietary restrictions
    DIETARY = {
        "vegetarian",
        "vegan",
        "gluten-free",
        "gluten free",
        "dairy-free",
        "dairy free",
        "nut-free",
        "nut free",
        "kosher",
        "halal",
        "pescatarian",
        "keto",
        "low-carb",
    }

    def __init__(self, menu_path: Optional[str] = None):
        """
        Initialize extractor.

        Args:
            menu_path: Path to menu.json file
        """
        self.menu_items: Dict[str, Dict] = {}
        self.menu_keywords: Set[str] = set()

        if menu_path:
            self.load_menu(menu_path)
        else:
            self._load_default_menu()

    def load_menu(self, menu_path: str):
        """Load menu items from JSON file."""
        path = Path(menu_path)
        if not path.exists():
            print(f"Menu file not found: {menu_path}")
            self._load_default_menu()
            return

        with open(path) as f:
            menu_data = json.load(f)

        self._process_menu_data(menu_data)

    def _load_default_menu(self):
        """Load default sample menu."""
        default_menu = {
            "categories": [
                {
                    "name": "Appetizers",
                    "items": [
                        {"name": "Caesar Salad", "keywords": ["caesar", "salad"]},
                        {"name": "Soup of the Day", "keywords": ["soup"]},
                        {"name": "Bruschetta", "keywords": ["bruschetta"]},
                        {"name": "Calamari", "keywords": ["calamari", "fried calamari"]},
                    ],
                },
                {
                    "name": "Entrees",
                    "items": [
                        {"name": "Grilled Salmon", "keywords": ["salmon", "fish"]},
                        {"name": "Ribeye Steak", "keywords": ["steak", "ribeye", "beef"]},
                        {"name": "Chicken Parmesan", "keywords": ["chicken", "parmesan"]},
                        {"name": "Pasta Primavera", "keywords": ["pasta", "primavera"]},
                        {"name": "Lobster", "keywords": ["lobster"]},
                        {"name": "Lamb Chops", "keywords": ["lamb"]},
                    ],
                },
                {
                    "name": "Sides",
                    "items": [
                        {"name": "French Fries", "keywords": ["fries", "french fries"]},
                        {"name": "Mashed Potatoes", "keywords": ["mashed potatoes", "potatoes"]},
                        {"name": "Steamed Vegetables", "keywords": ["vegetables", "veggies"]},
                        {"name": "Rice", "keywords": ["rice"]},
                    ],
                },
                {
                    "name": "Desserts",
                    "items": [
                        {"name": "Chocolate Cake", "keywords": ["chocolate cake", "cake"]},
                        {"name": "Cheesecake", "keywords": ["cheesecake"]},
                        {"name": "Ice Cream", "keywords": ["ice cream"]},
                        {"name": "Tiramisu", "keywords": ["tiramisu"]},
                    ],
                },
                {
                    "name": "Beverages",
                    "items": [
                        {"name": "Coffee", "keywords": ["coffee"]},
                        {"name": "Tea", "keywords": ["tea"]},
                        {"name": "Soda", "keywords": ["soda", "coke", "pepsi", "sprite"]},
                        {"name": "Water", "keywords": ["water"]},
                        {"name": "Wine", "keywords": ["wine", "red wine", "white wine"]},
                        {"name": "Beer", "keywords": ["beer"]},
                        {"name": "Juice", "keywords": ["juice", "orange juice"]},
                    ],
                },
            ]
        }
        self._process_menu_data(default_menu)

    def _process_menu_data(self, menu_data: Dict):
        """Process menu data into searchable format."""
        for category in menu_data.get("categories", []):
            for item in category.get("items", []):
                name = item["name"].lower()
                self.menu_items[name] = {
                    "name": item["name"],
                    "category": category["name"],
                    "keywords": [kw.lower() for kw in item.get("keywords", [])],
                }

                # Add keywords to search set
                self.menu_keywords.add(name)
                for kw in item.get("keywords", []):
                    self.menu_keywords.add(kw.lower())

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            ExtractionResult with extracted entities
        """
        entities = []
        text_lower = text.lower()

        # Extract quantities
        entities.extend(self._extract_quantities(text_lower))

        # Extract menu items
        entities.extend(self._extract_menu_items(text_lower))

        # Extract modifiers
        entities.extend(self._extract_modifiers(text_lower))

        # Extract dietary restrictions
        entities.extend(self._extract_dietary(text_lower))

        # Extract temperature
        entities.extend(self._extract_temperature(text_lower))

        # Extract size
        entities.extend(self._extract_size(text_lower))

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return ExtractionResult(
            entities=entities,
            text=text,
        )

    def _extract_quantities(self, text: str) -> List[Entity]:
        """Extract quantity entities."""
        entities = []

        # Number words
        for word, value in self.NUMBER_WORDS.items():
            pattern = rf"\b{word}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        type=EntityType.QUANTITY,
                        value=str(value),
                        original_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                    )
                )

        # Numeric quantities
        for match in re.finditer(r"\b(\d+)\b", text):
            value = int(match.group(1))
            if 1 <= value <= 20:  # Reasonable quantity range
                entities.append(
                    Entity(
                        type=EntityType.QUANTITY,
                        value=str(value),
                        original_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    )
                )

        return entities

    def _extract_menu_items(self, text: str) -> List[Entity]:
        """Extract menu item entities."""
        entities = []

        for keyword in self.menu_keywords:
            pattern = rf"\b{re.escape(keyword)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Find the full menu item name
                item_name = keyword
                for name, item_data in self.menu_items.items():
                    if keyword in item_data["keywords"] or keyword == name:
                        item_name = item_data["name"]
                        break

                entities.append(
                    Entity(
                        type=EntityType.MENU_ITEM,
                        value=item_name,
                        original_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                        metadata={"keyword": keyword},
                    )
                )

        return entities

    def _extract_modifiers(self, text: str) -> List[Entity]:
        """Extract cooking/preparation modifiers."""
        entities = []

        for modifier in self.MODIFIERS:
            pattern = rf"\b{re.escape(modifier)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        type=EntityType.MODIFIER,
                        value=modifier,
                        original_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                    )
                )

        return entities

    def _extract_dietary(self, text: str) -> List[Entity]:
        """Extract dietary restriction entities."""
        entities = []

        for dietary in self.DIETARY:
            pattern = rf"\b{re.escape(dietary)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        type=EntityType.DIETARY,
                        value=dietary,
                        original_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    )
                )

        return entities

    def _extract_temperature(self, text: str) -> List[Entity]:
        """Extract temperature preference entities."""
        entities = []

        for temp in self.TEMPERATURE:
            pattern = rf"\b{re.escape(temp)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        type=EntityType.TEMPERATURE,
                        value=temp,
                        original_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                    )
                )

        return entities

    def _extract_size(self, text: str) -> List[Entity]:
        """Extract size preference entities."""
        entities = []

        for size in self.SIZE:
            pattern = rf"\b{re.escape(size)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        type=EntityType.SIZE,
                        value=size,
                        original_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                    )
                )

        return entities

    def parse_order(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse an order from text into structured items.

        Args:
            text: Order text

        Returns:
            List of order items with quantity, item, and modifiers
        """
        result = self.extract(text)
        orders = []

        # Group entities into order items
        menu_items = result.get_entities_by_type(EntityType.MENU_ITEM)
        quantities = result.get_entities_by_type(EntityType.QUANTITY)
        modifiers = result.get_entities_by_type(EntityType.MODIFIER)

        for item in menu_items:
            order_item = {
                "item": item.value,
                "quantity": 1,
                "modifiers": [],
                "original_text": item.original_text,
            }

            # Find associated quantity (closest before the item)
            for qty in quantities:
                if qty.end <= item.start and item.start - qty.end < 20:
                    order_item["quantity"] = int(qty.value)
                    break

            # Find associated modifiers (within range of item)
            for mod in modifiers:
                if abs(mod.start - item.start) < 30 or abs(mod.end - item.end) < 30:
                    order_item["modifiers"].append(mod.value)

            orders.append(order_item)

        return orders
