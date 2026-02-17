import torch
import re

class SafetyFilter:
    def __init__(self, blocked_keywords=None):
        self.blocked_keywords = blocked_keywords or [
            "harmful", "illegal", "toxic", "offensive"
        ]
        
    def check_text(self, text):
        """Simple keyword-based safety check."""
        for word in self.blocked_keywords:
            if re.search(rf"\b{word}\b", text, re.IGNORECASE):
                return False, f"Content contains blocked keyword: {word}"
        return True, None

    def validate_function_call(self, name, args, allowed_functions):
        """Validate if the function call is permitted."""
        if name not in allowed_functions:
            return False, f"Function '{name}' is not in the allowed list."
        return True, None

def apply_safety_guard(response, safety_filter):
    is_safe, reason = safety_filter.check_text(response)
    if not is_safe:
        return f"[Blocked by Safety Filter: {reason}]"
    return response
