"""Cut a reply into sentences, from a finished string or a token stream.

Replies are always spoken one sentence at a time: the first sentence goes to
TTS while gemma4 is still writing the rest and the next one is synthesized
while the first plays. gemma4 writes ~108 tokens/s on an M5 Max, ~15x faster
than speech, so its first sentence is done after 0.2-0.4 s (measured Oct 2026)
and TTS, not the LLM, sets the pace.
"""

import re

# A sentence ends at . ! ? or … followed by whitespace and then something that
# can start a sentence. Waiting for that next character costs one token but
# keeps "t.ex. Plopp" and "kl. 3" in one piece.
_BOUNDARY = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…][\"”»)]))\s+(?=[\"“«(]?[A-ZÅÄÖÆØÉÜ0-9¿¡])")

# Emojis and pictographs: TTS reads them aloud or chokes on them.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\u2B00-\u2BFF]+")
_MARKDOWN = re.compile(r"[*_#`]+")

# A period after one of these does not end the sentence ("t.ex. Plopp").
_ABBREV = re.compile(
    r"(?:^|\s)(?:t\.ex|bl\.a|p\.g\.a|f\.d|d\.v\.s|kl|ca|nr|resp|"
    r"e\.g|i\.e|mr|mrs|ms|dr|st)\.$", re.IGNORECASE)


def clean(text: str) -> str:
    """Strip what should never be spoken: emojis, markdown marks."""
    return " ".join(_MARKDOWN.sub("", _EMOJI.sub("", text)).split())


def _cut(text: str) -> list:
    """Split at sentence ends, gluing back the ones after an abbreviation."""
    parts = []
    for p in _BOUNDARY.split(_EMOJI.sub("", text)):
        if parts and _ABBREV.search(parts[-1]):
            parts[-1] += " " + p
        else:
            parts.append(p)
    return parts


def split_sentences(text: str) -> list:
    """Sentences of a finished text, cleaned, empty ones dropped."""
    return [s for s in (clean(p) for p in _cut(text or "")) if s]


class SentenceStream:
    """Feed tokens in, get finished sentences out.

    ``feed()`` returns the sentences the new text completed; ``flush()``
    returns whatever is left once the stream ends.
    """

    def __init__(self):
        self._buf = ""

    def feed(self, text: str) -> list:
        self._buf += text
        parts = _cut(self._buf)
        self._buf = parts.pop()   # the last part may still grow
        return [s for s in (clean(p) for p in parts) if s]

    def flush(self) -> list:
        rest, self._buf = self._buf, ""
        return split_sentences(rest)
