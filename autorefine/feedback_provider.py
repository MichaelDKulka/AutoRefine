"""Abstract feedback provider for collecting user feedback strings.

Developers subclass :class:`FeedbackProvider` to define *how* their
application collects feedback from end-users.  AutoRefine calls the
provider after each response, and the returned string is recorded as
feedback that drives prompt refinement.

Example — CLI prompt::

    from autorefine import FeedbackProvider

    class CLIFeedback(FeedbackProvider):
        def get_feedback(self, response_id: str, response_text: str) -> str:
            return input("How was this response? (or press Enter to skip): ")

Example — web callback::

    class WebhookFeedback(FeedbackProvider):
        def __init__(self, queue):
            self._queue = queue

        def get_feedback(self, response_id: str, response_text: str) -> str:
            # Block until the frontend posts feedback for this response
            return self._queue.get(response_id, timeout=300)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class FeedbackProvider(ABC):
    """Abstract base class for collecting feedback strings from end-users.

    Subclass this and implement :meth:`get_feedback` to plug in your own
    feedback collection mechanism (CLI input, web form, Slack bot, etc.).

    The returned string is the user's raw feedback on what the AI provided.
    AutoRefine analyses the sentiment automatically and uses it to refine
    the prompt once enough feedback (default: 10) has accumulated.
    """

    @abstractmethod
    def get_feedback(self, response_id: str, response_text: str) -> str:
        """Collect a feedback string from the end-user.

        Args:
            response_id: The unique ID of the AI response (from
                :attr:`CompletionResponse.id`).
            response_text: The AI's response text, so the provider can
                display it alongside the feedback prompt if needed.

        Returns:
            The user's feedback as a free-text string.  Return an empty
            string to skip feedback for this response (no signal recorded).
        """
        ...

    def classify(self, feedback_text: str) -> str:
        """Classify feedback text as positive or negative.

        Override this to provide custom sentiment classification.  The
        default implementation uses simple keyword matching.

        Args:
            feedback_text: The raw feedback string from the user.

        Returns:
            ``"positive"`` or ``"negative"``.
        """
        text = feedback_text.lower().strip()
        positive_signals = [
            "good", "great", "perfect", "thanks", "thank", "helpful",
            "nice", "awesome", "excellent", "correct", "right", "yes",
            "love", "amazing", "well done", "spot on", "accurate",
        ]
        if any(word in text for word in positive_signals):
            return "positive"
        return "negative"
