"""Intent classification for customer requests."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re


class Intent(str, Enum):
    """Customer intent categories."""

    ORDER_FOOD = "order_food"
    ORDER_DRINK = "order_drink"
    REQUEST_CHECK = "request_check"
    REQUEST_REFILL = "request_refill"
    ASK_QUESTION = "ask_question"
    MAKE_COMPLAINT = "make_complaint"
    GIVE_COMPLIMENT = "give_compliment"
    REQUEST_SERVICE = "request_service"
    REQUEST_MENU = "request_menu"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: Intent
    confidence: float
    sub_intents: List[Intent] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "sub_intents": [i.value for i in self.sub_intents],
            "reasoning": self.reasoning,
        }


class IntentClassifier:
    """
    Classifies customer intent from transcribed text.

    Uses a combination of:
    - Keyword matching
    - Pattern matching
    - Optional: sentence-transformers for semantic similarity
    """

    # Intent patterns (regex)
    INTENT_PATTERNS = {
        Intent.ORDER_FOOD: [
            r"\b(order|have|get|try|want)\b.*\b(food|dish|meal|appetizer|entree|dessert)\b",
            r"\bi'?ll?\s+(have|take|get)\b",
            r"\bcan\s+i\s+(have|get|order)\b.*\b(the|a|an)\b",
            r"\bi'?d?\s+like\s+(to\s+)?(order|have|try)\b",
        ],
        Intent.ORDER_DRINK: [
            r"\b(order|have|get|want)\b.*\b(drink|water|coffee|tea|wine|beer|cocktail|juice|soda)\b",
            r"\bcan\s+i\s+get\b.*\b(water|drink|coffee|tea)\b",
            r"\bi'?ll?\s+(have|take)\b.*\b(to\s+drink)\b",
        ],
        Intent.REQUEST_CHECK: [
            r"\b(check|bill|tab)\b.*\bplease\b",
            r"\bcan\s+(i|we)\s+(have|get)\s+(the\s+)?(check|bill)\b",
            r"\bready\s+to\s+pay\b",
            r"\bwe'?re?\s+done\b",
            r"\bpay\s+(the\s+)?(bill|check)\b",
        ],
        Intent.REQUEST_REFILL: [
            r"\b(refill|more|another)\b.*\b(water|coffee|drink|tea)\b",
            r"\bcan\s+(i|we)\s+(have|get)\s+(a\s+)?(refill|more)\b",
            r"\bmy\s+(glass|cup)\s+is\s+empty\b",
        ],
        Intent.ASK_QUESTION: [
            r"\bwhat\s+(is|are|do|does|would)\b",
            r"\bhow\s+(is|do|does|long|much)\b",
            r"\bis\s+(this|that|there|it)\b.*\?",
            r"\bdo\s+you\s+(have|serve|recommend)\b",
            r"\bwhat'?s?\s+in\s+(the|this)\b",
        ],
        Intent.MAKE_COMPLAINT: [
            r"\b(cold|raw|undercooked|overcooked|burnt|wrong)\b",
            r"\b(waiting|waited)\s+(too\s+)?long\b",
            r"\bthis\s+is(n'?t)?\s+(right|wrong|correct)\b",
            r"\b(terrible|awful|bad|horrible|disgusting)\b",
            r"\bnot\s+what\s+i\s+ordered\b",
            r"\bwhere\s+is\s+(my|our)\b",
        ],
        Intent.GIVE_COMPLIMENT: [
            r"\b(delicious|excellent|amazing|wonderful|great|fantastic|perfect)\b",
            r"\bthis\s+is\s+(so\s+)?(good|great|amazing)\b",
            r"\bcompliments\s+to\b",
            r"\breally\s+(enjoy|love|like)\b",
        ],
        Intent.REQUEST_SERVICE: [
            r"\bexcuse\s+me\b",
            r"\bcan\s+(i|we)\s+get\s+(some\s+)?help\b",
            r"\bwaiter|waitress|server\b",
            r"\bover\s+here\b",
        ],
        Intent.REQUEST_MENU: [
            r"\b(see|have|get)\s+(the\s+)?(menu|dessert\s+menu|wine\s+list)\b",
            r"\bmenu\s+please\b",
        ],
        Intent.GREETING: [
            r"\b(hello|hi|hey|good\s+(morning|afternoon|evening))\b",
            r"\bhow\s+are\s+you\b",
        ],
    }

    # Keywords for quick matching
    INTENT_KEYWORDS = {
        Intent.ORDER_FOOD: ["order", "have", "get", "like", "try", "eat"],
        Intent.ORDER_DRINK: ["drink", "water", "coffee", "tea", "wine", "beer", "cocktail"],
        Intent.REQUEST_CHECK: ["check", "bill", "tab", "pay", "done", "finished"],
        Intent.REQUEST_REFILL: ["refill", "more", "another", "empty"],
        Intent.ASK_QUESTION: ["what", "how", "when", "where", "which", "recommend"],
        Intent.MAKE_COMPLAINT: ["wrong", "cold", "waiting", "problem", "issue", "terrible"],
        Intent.GIVE_COMPLIMENT: ["delicious", "excellent", "amazing", "wonderful", "great"],
        Intent.REQUEST_SERVICE: ["excuse", "help", "waiter", "server", "please"],
        Intent.REQUEST_MENU: ["menu", "list"],
        Intent.GREETING: ["hello", "hi", "hey"],
    }

    def __init__(self, use_embeddings: bool = False):
        """
        Initialize classifier.

        Args:
            use_embeddings: Whether to use sentence-transformers for semantic matching
        """
        self.use_embeddings = use_embeddings
        self.embedding_model = None

        if use_embeddings:
            self._load_embedding_model()

    def _load_embedding_model(self):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Loaded sentence-transformers model")
        except ImportError:
            print("sentence-transformers not available, using pattern matching only")
            self.use_embeddings = False

    def classify(self, text: str) -> IntentResult:
        """
        Classify intent from text.

        Args:
            text: Input text to classify

        Returns:
            IntentResult with classified intent
        """
        text_lower = text.lower().strip()

        # Try pattern matching first
        pattern_result = self._classify_patterns(text_lower)
        if pattern_result.confidence >= 0.8:
            return pattern_result

        # Try keyword matching
        keyword_result = self._classify_keywords(text_lower)

        # Combine results
        if pattern_result.confidence > keyword_result.confidence:
            return pattern_result
        elif keyword_result.confidence > pattern_result.confidence:
            return keyword_result
        else:
            # Equal confidence - use keyword result as tiebreaker
            return keyword_result

    def _classify_patterns(self, text: str) -> IntentResult:
        """Classify using regex patterns."""
        best_intent = Intent.UNKNOWN
        best_confidence = 0.0
        matches = []

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append(intent)
                    confidence = 0.85  # High confidence for pattern match
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
                    break

        # Check for multiple intents
        sub_intents = [i for i in set(matches) if i != best_intent]

        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            sub_intents=sub_intents,
            reasoning=f"Pattern match: {len(matches)} patterns matched",
        )

    def _classify_keywords(self, text: str) -> IntentResult:
        """Classify using keyword matching."""
        scores: Dict[Intent, float] = {}
        words = set(text.split())

        for intent, keywords in self.INTENT_KEYWORDS.items():
            matched = sum(1 for kw in keywords if kw in text)
            if matched > 0:
                # Score based on keyword density
                scores[intent] = matched / len(keywords)

        if not scores:
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.3,
                reasoning="No keywords matched",
            )

        # Get best intent
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # Convert to confidence (0-1)
        confidence = min(0.7, best_score * 2)  # Cap at 0.7 for keyword match

        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            sub_intents=[i for i in scores if i != best_intent and scores[i] > 0.3],
            reasoning=f"Keyword match: {best_score:.2f} score",
        )

    def classify_with_context(
        self,
        text: str,
        table_state: str,
        previous_intents: Optional[List[Intent]] = None,
    ) -> IntentResult:
        """
        Classify with additional context.

        Args:
            text: Input text
            table_state: Current state of the table
            previous_intents: Previous classified intents

        Returns:
            IntentResult with context-aware classification
        """
        # Base classification
        result = self.classify(text)

        # Adjust based on table state
        state_adjustments = {
            "seated": {Intent.ORDER_DRINK: 0.1, Intent.REQUEST_MENU: 0.1},
            "ordering": {Intent.ORDER_FOOD: 0.1, Intent.ASK_QUESTION: 0.05},
            "waiting": {Intent.ASK_QUESTION: 0.05, Intent.MAKE_COMPLAINT: 0.05},
            "served": {Intent.GIVE_COMPLIMENT: 0.05, Intent.REQUEST_CHECK: 0.1},
            "paying": {Intent.REQUEST_CHECK: 0.1},
        }

        adjustments = state_adjustments.get(table_state, {})
        if result.intent in adjustments:
            result.confidence = min(1.0, result.confidence + adjustments[result.intent])
            result.metadata["state_adjusted"] = True

        return result

    def get_intent_description(self, intent: Intent) -> str:
        """Get human-readable description of intent."""
        descriptions = {
            Intent.ORDER_FOOD: "Customer wants to order food",
            Intent.ORDER_DRINK: "Customer wants to order a drink",
            Intent.REQUEST_CHECK: "Customer wants the check/bill",
            Intent.REQUEST_REFILL: "Customer wants a refill",
            Intent.ASK_QUESTION: "Customer has a question",
            Intent.MAKE_COMPLAINT: "Customer is making a complaint",
            Intent.GIVE_COMPLIMENT: "Customer is giving a compliment",
            Intent.REQUEST_SERVICE: "Customer needs service attention",
            Intent.REQUEST_MENU: "Customer wants to see the menu",
            Intent.GREETING: "Customer greeting",
            Intent.UNKNOWN: "Intent could not be determined",
        }
        return descriptions.get(intent, "Unknown intent")
