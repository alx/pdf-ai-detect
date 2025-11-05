"""
AI Text Detection Module using Fast-DetectGPT approach
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastDetectGPTDetector:
    """
    Wrapper for Fast-DetectGPT algorithm to detect AI-generated text.
    Uses conditional probability curvature to determine if text is machine-generated.
    """

    def __init__(
        self,
        scoring_model_name: str = "gpt2",
        sampling_model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the detector with scoring and sampling models.

        Args:
            scoring_model_name: Name of the model to use for scoring (default: gpt2)
            sampling_model_name: Name of the model to use for sampling (default: same as scoring)
            device: Device to run models on (default: auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load scoring model
        logger.info(f"Loading scoring model: {scoring_model_name}")
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
        self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model_name)
        self.scoring_model.to(self.device)
        self.scoring_model.eval()

        # For simplicity, use the same model for sampling if not specified
        if sampling_model_name and sampling_model_name != scoring_model_name:
            logger.info(f"Loading sampling model: {sampling_model_name}")
            self.sampling_tokenizer = AutoTokenizer.from_pretrained(sampling_model_name)
            self.sampling_model = AutoModelForCausalLM.from_pretrained(sampling_model_name)
            self.sampling_model.to(self.device)
            self.sampling_model.eval()
        else:
            self.sampling_tokenizer = self.scoring_tokenizer
            self.sampling_model = self.scoring_model

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        if self.sampling_tokenizer.pad_token is None:
            self.sampling_tokenizer.pad_token = self.sampling_tokenizer.eos_token

    def get_log_likelihood(self, text: str) -> float:
        """
        Calculate the log-likelihood of the text under the scoring model.

        Args:
            text: Input text to score

        Returns:
            Log-likelihood score
        """
        with torch.no_grad():
            inputs = self.scoring_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            outputs = self.scoring_model(**inputs, labels=inputs["input_ids"])
            return -outputs.loss.item()

    def detect(self, text: str, return_prob: bool = True) -> Dict[str, float]:
        """
        Detect if text is AI-generated using Fast-DetectGPT approach.

        Args:
            text: Text to analyze
            return_prob: Whether to return probability estimate

        Returns:
            Dictionary containing:
                - score: Detection score (higher = more likely AI-generated)
                - probability: Estimated probability of being AI-generated (if return_prob=True)
                - log_likelihood: Log-likelihood under the scoring model
        """
        if not text or len(text.strip()) < 10:
            # Too short to analyze reliably
            return {
                "score": 0.0,
                "probability": 0.5,
                "log_likelihood": 0.0,
            }

        try:
            # Get log-likelihood of the original text
            log_likelihood = self.get_log_likelihood(text)

            # Simple heuristic: higher log-likelihood suggests more predictable (potentially AI-generated) text
            # Normalize the score to a reasonable range
            # This is a simplified version - the full Fast-DetectGPT uses conditional probability curvature
            score = -log_likelihood  # Invert so higher score = more AI-like

            # Convert score to probability estimate (sigmoid-like transformation)
            # These constants are heuristic and would need tuning on real data
            probability = 1 / (1 + np.exp(-score))

            return {
                "score": float(score),
                "probability": float(probability),
                "log_likelihood": float(log_likelihood),
            }

        except Exception as e:
            logger.error(f"Error detecting text: {e}")
            return {
                "score": 0.0,
                "probability": 0.5,
                "log_likelihood": 0.0,
            }

    def score_text(self, text: str) -> float:
        """
        Get a simple score for text (0.0 to 1.0, where 1.0 is most likely AI-generated).

        Args:
            text: Text to score

        Returns:
            Score between 0.0 and 1.0
        """
        result = self.detect(text)
        return result["probability"]


# Lightweight detector for faster processing
class SimpleAIDetector:
    """
    Simplified AI detector that uses basic perplexity scoring.
    Faster than FastDetectGPTDetector but less accurate.
    """

    def __init__(self, model_name: str = "gpt2"):
        """Initialize with a language model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading simple detector with model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def score_text(self, text: str) -> float:
        """
        Score text based on perplexity (lower perplexity = more likely AI-generated).

        Args:
            text: Text to score

        Returns:
            Score between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 10:
            return 0.5

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                perplexity = torch.exp(outputs.loss).item()

                # Lower perplexity = more predictable = more likely AI-generated
                # Map perplexity to 0-1 score (these thresholds are heuristic)
                # Typical perplexity ranges: AI-generated ~10-50, human ~50-200
                if perplexity < 20:
                    score = 0.9
                elif perplexity < 50:
                    score = 0.7
                elif perplexity < 100:
                    score = 0.4
                else:
                    score = 0.2

                return float(score)

        except Exception as e:
            logger.error(f"Error scoring text: {e}")
            return 0.5
