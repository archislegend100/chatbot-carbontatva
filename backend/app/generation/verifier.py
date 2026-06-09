import logging
from typing import List, Dict, Any
from app.config.settings import settings
from app.routing.query_router import QueryProfile
from app.generation.mistral_client import mistral_client
from app.generation.prompts import VERIFICATION_PROMPT

logger = logging.getLogger(__name__)

class AnswerVerifier:
    def __init__(self):
        self.enabled = settings.ENABLE_VERIFICATION

    async def verify(self, query: str, proposed_answer: str, context_blocks: str, query_profile: QueryProfile) -> str:
        """
        Runs a lightweight verification pass for hard/technical queries.
        If verification fails, returns a revised answer. Otherwise returns original.
        """
        if not self.enabled or not query_profile.needs_verification:
            return proposed_answer

        prompt = VERIFICATION_PROMPT.format(
            context_blocks=context_blocks,
            proposed_answer=proposed_answer
        )
        
        try:
            # We can use a fast model if available, but for now reuse the main one
            verification_result = await mistral_client.generate(prompt)
            
            verification_result = verification_result.strip()
            if verification_result == "PASS" or "PASS" in verification_result[:10]:
                logger.info("Answer passed verification.")
                return proposed_answer
            else:
                logger.warning("Answer failed verification. Using revised answer.")
                return verification_result
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return proposed_answer # Fall back to original on API error

verifier = AnswerVerifier()
