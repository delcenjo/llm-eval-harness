from .config import CHAT_MODEL, LABELS


class KeywordBaseline:
    name = "keyword-baseline"

    RULES = {
        "billing": ["invoice", "charge", "charged", "payment", "refund", "price", "bill", "receipt", "credit card"],
        "technical": ["error", "bug", "crash", "freeze", "not working", "loading", "slow", "broken"],
        "account": ["password", "username", "email address", "profile", "settings", "2fa", "two-factor", "reset", "log in", "login"],
        "cancellation": ["cancel", "unsubscribe", "delete my account", "close my account", "terminate"],
    }

    def predict(self, text):
        lowered = text.lower()
        for label, keywords in self.RULES.items():
            if any(keyword in lowered for keyword in keywords):
                return label
        return "other"


class ClaudeClassifier:
    def __init__(self, model=CHAT_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"claude:{model}"

    def predict(self, text):
        prompt = (
            f"Classify the support message into exactly one of these categories: "
            f"{', '.join(LABELS)}.\nReply with the single category word only.\n\n"
            f"Message: {text}"
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = message.content[0].text.strip().lower()
        return next((label for label in LABELS if label in answer), "other")
