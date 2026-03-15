# Custom Refiners

By default, AutoRefine uses Claude as the refiner. You can customize the refinement logic in several ways.

## Use a different refiner model

```python
client = AutoRefine(
    api_key="sk-...",
    model="gpt-4o",
    refiner_key="sk-...",           # can be same key as primary
    refiner_model="gpt-4o",         # use GPT-4o as refiner
    refiner_provider="openai",
)
```

## Use a local model (Ollama)

```python
client = AutoRefine(
    api_key="sk-...",
    model="gpt-4o",
    refiner_key="not-needed",
    refiner_model="llama3.1",
    refiner_provider="ollama",
)
```

!!! warning
    Local models produce lower-quality refinements than Claude or GPT-4o.
    Use for experimentation, not production.

## Write a custom provider

Implement `BaseProvider` to use any model as the refiner:

```python
from autorefine.providers.base import BaseProvider, ProviderResponse
from autorefine.models import Message

class MyRefinerProvider(BaseProvider):
    name = "my-refiner"

    def chat(self, system_prompt: str, messages: list[Message], **kwargs) -> ProviderResponse:
        # Your custom logic here
        prompt = messages[-1].content  # the meta-prompt
        improved = my_custom_refinement(prompt)
        return ProviderResponse(text=improved, input_tokens=0, output_tokens=0)

    def stream(self, system_prompt, messages, **kwargs):
        yield self.chat(system_prompt, messages).text

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.0
```

Wire it in:

```python
from autorefine import AutoRefine
from autorefine.refiner import Refiner

client = AutoRefine(api_key="sk-...", model="gpt-4o", store=my_store)
client._refiner = Refiner(
    refiner_provider=MyRefinerProvider(),
    store=client._store,
    prompt_key="default",
)
```

## Control the meta-prompt

The meta-prompt (in `autorefine/refiner.py`) is the most critical piece.
You can customize it by subclassing `Refiner`:

```python
from autorefine.refiner import Refiner, META_PROMPT

class MyRefiner(Refiner):
    def _call_refiner(self, current_prompt, version, bundles, feedback):
        # Add domain-specific instructions
        custom_instructions = """
        IMPORTANT: This is a medical chatbot. Never suggest treatments.
        Only improve how the bot communicates existing approved responses.
        """
        # Modify the meta-prompt
        from autorefine.refiner import _build_interaction_log, _build_feedback_summary, REFINER_SYSTEM_PROMPT
        from autorefine.models import CostEntry

        interaction_log = _build_interaction_log(bundles, scrubber=self._scrubber)
        feedback_summary = _build_feedback_summary(feedback)

        meta = META_PROMPT.format(
            current_prompt=current_prompt,
            prompt_version=version,
            interaction_log=interaction_log,
            feedback_summary=feedback_summary,
        ) + custom_instructions

        resp = self._provider.complete(REFINER_SYSTEM_PROMPT, meta)
        cost = self._provider.estimate_cost(resp.input_tokens, resp.output_tokens)
        try:
            self._store.save_cost_entry(CostEntry(
                model=resp.model, provider=self._provider.name,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                cost_usd=cost, call_type="refiner",
            ))
        except Exception:
            pass
        return self._parse_refiner_response(resp.text, feedback_summary)
```

## Manual refinement

Skip auto-refinement entirely and trigger manually:

```python
client = AutoRefine(
    api_key="sk-...", model="gpt-4o",
    refiner_key="sk-ant-...",
    auto_learn=False,  # don't auto-trigger
)

# ... collect feedback ...

# Trigger when you're ready
version = client.refine_now()
if version:
    print(f"Refined to v{version.version}: {version.system_prompt[:80]}...")
```
