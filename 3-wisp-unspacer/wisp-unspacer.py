import regex

class WISPunspacer:
    """
    Create reversible whitespace-removed variants of poems - transforming whitespace-rich
    poems to linear prose-like form. This involves representing the original poem as a sequence
    of contiguous whitespace and non-whitespace units, such that the whitespace units can be 
    replaced by a single space for NLP analysis. The original poem can be reconstructed back
    from this representation by swapping the whitespace units back in for single spaces.
      1. Parse the poem into alternating [text, whitespace, text, whitespace, ...] units.
      2. Creates a "collapsed" prose version, where multi-spaces/tabs become one space.
      3. Stores metadata about whitespace runs.
      4. Reconstruct the original poem faithfully by swapping back the original whitespace strings
        into their initial positions between specific non-whitespace units.

    WISP LABELS: Each whitespace unit ("ws") is additionally annotated with:
        - wisp_labels: list (alphabetically sorted) of zero or more labels among:
                * "internal"    -> ws contains NO '\n' and is more than a single ordinary ASCII space OR contains 
                                   a tab OR contains any other Unicode space character with more than single-space width
                * "line_break"  -> any ws containing at least one '\n'
                * "prefix"      -> ws contains a '\n' and has trailing horizontal indentation (spaces / tabs / Unicode separators) 
                                   after the LAST newline.
                * "vertical"    -> ws contains two or more '\n' (>=2)
            Rules:
                - 'internal' never appears with newline-based labels.
                - 'vertical' implies 'line_break'.
                - 'prefix' can co-exist with 'vertical'.
        - newline_count: count of '\n' characters in ws.

    NOTE: A contiguous pair of text unit + following whitespace unit is called a "chunk".
    """
    def __init__(self, poem: str, ordinary_singletons=None):
        self.original_poem = poem
        # ordinary singleton whitespace codepoints that, when appearing ALONE (length==1),
        # should NOT be considered 'internal'. This is configurable.
        # Default chosen: ASCII space, NBSP, NARROW NO-BREAK SPACE, THIN SPACE, HAIR SPACE.
        # (omitting wider spaces like EM/EN/IDEOGRAPHIC to treat them as stylistically significant.)
        if ordinary_singletons is None:
            ordinary_singletons = {" ", "\u00A0", "\u202F", "\u2009", "\u200A"}
        self.ordinary_singletons = ordinary_singletons
        # store each segment as a dict of:
        #   { "text": <string of non-whitespace>,
        #     "ws": <string of whitespace that follows>,
        #     "wisp_labels": [...],      # alphabetically sorted
        #     "newline_count": int }
        self.chunks = self._parse_poem(poem)
        self.prose = self._to_prose(self.chunks)
        self.reconstructed_poem = self._reconstruct(self.chunks)
        self.reversible = (self.original_poem == self.reconstructed_poem)

    def _parse_poem(self, poem: str):
        """
        Return a list of dicts, each with:
          text -> non-whitespace substring
          ws   -> the exact whitespace that followed that text
          wisp_labels -> list of zero or more labels among ["internal", "line_break", "prefix", "vertical"]
          newline_count -> count of '\n' in ws

        "Hello\n  world!", tokens from regex might be ["Hello", "\n  ", "world!"].
        This method turns that into:
          [ 
            { "text": "Hello", "ws": "\n  ", "wisp_labels": ["line_break", "prefix"], "newline_count": 1 },
            { "text": "world!", "ws": "",     "wisp_labels": [], "newline_count": 0 }
          ]
        Addendum: I merge any empty 'text' chunks (except at the very beginning)
        into the previous chunk's 'ws', to avoid getting a middle chunk with text=""
        """
        # split on runs of whitespace *keeping* them (due to parentheses)
        # using regex's `\p{Z}` to match additional UNICODE separators
        tokens = regex.split(r'([\p{Z}\s]+)', poem)
        chunks = []
        for i in range(0, len(tokens), 2):
            text_chunk = tokens[i]
            ws_chunk = tokens[i+1] if i+1 < len(tokens) else ""
            if text_chunk or not chunks:
                labels, newline_count = self._classify_whitespace(ws_chunk)
                chunk = {
                    "text": text_chunk,
                    "ws": ws_chunk,
                    "wisp_labels": labels,
                    "newline_count": newline_count,
                }
                chunks.append(chunk)
            else:
                # merge empty text into previous whitespace
                prev = chunks[-1]
                prev_ws = prev["ws"] + ws_chunk
                labels, newline_count = self._classify_whitespace(prev_ws)
                prev["ws"] = prev_ws
                prev["wisp_labels"] = labels
                prev["newline_count"] = newline_count
        return chunks

    def _classify_whitespace(self, ws: str):
        """Return (labels, newline_count) for a whitespace string.

        Adjusted rules:
          - A single ASCII space OR a single NBSP (\u00A0) -> no labels.
          - INTERNAL if no newlines and NOT (single ASCII space or single NBSP).
          - LINE_BREAK if >=1 '\n'. VERTICAL if >=2 '\n'. PREFIX if any horizontal chars after last '\n'.
          - Labels sorted alphabetically before returning.
        """
        if not ws:
            return [], 0
        newline_count = ws.count('\n')
        labels = []
        if newline_count > 0:
            labels.append("line_break")
            if newline_count >= 2:
                labels.append("vertical")
            trailing = ws[ws.rfind('\n')+1:]
            if trailing:
                labels.append("prefix")
        else:
            # no newlines; determine internal via singleton set
            if not (len(ws) == 1 and ws in self.ordinary_singletons):
                labels.append("internal")
        labels.sort()
        return labels, newline_count

    def _to_prose(self, chunks):
        # collapses each whitespace run to a single space (for easier NLP)
        prose_parts = []
        for i, chunk in enumerate(chunks):
            text, ws = chunk["text"], chunk["ws"]
            # add the text directly
            prose_parts.append(text)
            # if the whitespace run is non-empty and this is not the last chunk,
            # I want exactly ONE space to separate from the next text
            if ws and i < len(chunks) - 1:
                prose_parts.append(" ")
        # join them all and strip once (to avoid leading/trailing spaces):
        return "".join(prose_parts).strip()

    def _reconstruct(self, chunks):
        # join all text + the original whitespace to fully recreate the poem
        out = []
        for chunk in chunks:
            out.append(chunk["text"])
            out.append(chunk["ws"])
        return "".join(out)

    def show_chunks(self):
        for ind, chunk in enumerate(self.chunks):
            print(
                f"Chunk {ind}: text={repr(chunk['text'])}, whitespace={repr(chunk['ws'])}, "
                f"wisp_labels={chunk['wisp_labels']}, newline_count={chunk['newline_count']}"
            )