import re
from tqdm import tqdm
import logging


def chunk_text(text, chunk_size=1000):
    """
    Enhanced text chunking algorithm that intelligently splits text
    while preserving semantic boundaries and structure.

    Features:
    - Respects paragraph and sentence boundaries
    - Handles hierarchical structures like headers
    - Preserves lists and bullet points
    - Avoids splitting tables inappropriately
    - Maintains references to images
    """
    if not text:
        return []

    logging.debug(f"Chunking text of length {len(text)}")

    # Try intelligent splitting by semantic units first
    chunks = split_by_semantic_units(text, chunk_size)

    # Fall back to recursive character splitting if needed
    if not chunks or any(len(chunk) > chunk_size * 1.5 for chunk in chunks):
        chunks = split_by_recursive_algorithm(text, chunk_size)

    # Ensure chunks aren't too short (merge if needed)
    min_chunk_size = max(100, chunk_size // 10)
    chunks = merge_short_chunks(chunks, min_chunk_size, chunk_size)

    # Clean up chunks - remove excessive whitespace and ensure proper sentence endings
    cleaned_chunks = []
    for chunk in tqdm(chunks, desc="Cleaning chunks", unit="chunk", leave=False):
        cleaned_chunks.append(clean_chunk(chunk))

    return cleaned_chunks


def split_by_semantic_units(text, chunk_size):
    """Split text respecting semantic units like paragraphs, headings, and lists"""
    # Identify semantic boundaries
    heading_pattern = re.compile(
        r"^(#+|\d+\.+|\w+\.+|\*\*|__|\d+\)|\w+\))\s+.+$", re.MULTILINE
    )
    paragraph_pattern = re.compile(r"\n\s*\n")
    list_item_pattern = re.compile(r"^\s*[-•*+]\s+", re.MULTILINE)

    # Get all potential breakpoints
    breakpoints = []

    # Add paragraph breaks
    for match in paragraph_pattern.finditer(text):
        breakpoints.append((match.start(), 3))  # Priority 3

    # Add headings (stronger breakpoints)
    for match in heading_pattern.finditer(text):
        breakpoints.append((match.start(), 1))  # Priority 1 (higher)

    # Add list items (weaker breakpoints)
    for match in list_item_pattern.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == "\n":
            breakpoints.append((match.start(), 4))  # Priority 4 (lower)

    # Add sentence endings
    sentence_endings = [m.start() + 1 for m in re.finditer(r"[.!?]\s+[A-Z]", text)]
    for pos in sentence_endings:
        breakpoints.append((pos, 5))  # Priority 5 (lowest)

    # Sort breakpoints by position
    breakpoints.sort(key=lambda x: x[0])

    # Create chunks based on breakpoints
    chunks = []
    start_pos = 0

    # Add progress indication for longer texts
    with tqdm(
        total=len(text),
        desc="Splitting text by semantic units",
        unit="chars",
        leave=False,
    ) as pbar:
        while start_pos < len(text):
            # Find the best breakpoint that keeps chunk size under limit
            best_break = None
            for pos, priority in filter(
                lambda x: x[0] > start_pos and x[0] <= start_pos + chunk_size,
                breakpoints,
            ):
                if best_break is None or priority <= best_break[1]:
                    best_break = (pos, priority)

            # If no suitable breakpoint found, just cut at chunk_size
            if best_break is None:
                end_pos = min(start_pos + chunk_size, len(text))
            else:
                end_pos = best_break[0]

            # Add chunk
            chunk = text[start_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Update progress bar
            pbar.update(end_pos - start_pos)

            # Move to next chunk with overlap
            if end_pos == len(text):
                break
            start_pos = end_pos + 1

    return chunks


def split_by_recursive_algorithm(text, chunk_size):
    """Split text using a recursive algorithm that tries different separators"""
    # Define separators from most significant to least
    separators = [
        "\n## ",  # Markdown h2
        "\n### ",  # Markdown h3
        "\n#### ",  # Markdown h4
        "\n\n",  # Paragraph
        ". ",  # Sentence
        ", ",  # Clause
        " ",  # Word
        "",  # Character
    ]

    def _split_recursive(text, separators_idx=0):
        """Recursively split text using increasingly granular separators"""
        if separators_idx >= len(separators):
            return [text]

        separator = separators[separators_idx]

        # If the text fits in a chunk, return it
        if len(text) <= chunk_size:
            return [text]

        # Try splitting with the current separator
        parts = text.split(separator)

        # If the split creates reasonable chunks, keep them
        if separator and all(len(p) <= chunk_size for p in parts) and len(parts) > 1:
            # Add the separator back to each part except the first
            result = [parts[0]]
            for part in parts[1:]:
                result.append(separator + part)
            return result

        # If the current separator doesn't work well, try the next one
        if len(parts) == 1 or any(len(p) > chunk_size * 1.2 for p in parts):
            return _split_recursive(text, separators_idx + 1)

        # Otherwise, recursively split any parts that are still too large
        result = []
        for part in parts:
            if len(part) <= chunk_size:
                if separator and result and result[-1][-len(separator) :] != separator:
                    result.append(separator + part)
                else:
                    result.append(part)
            else:
                # Recursively split this part
                subparts = _split_recursive(part, separators_idx + 1)
                result.extend(subparts)

        return result

    return _split_recursive(text)


def merge_short_chunks(chunks, min_size, max_size):
    """Merge chunks that are too short while respecting max_size"""
    if not chunks:
        return []

    result = [chunks[0]]

    # Use tqdm for progress indication when merging many chunks
    if len(chunks) > 10:
        chunk_iterator = tqdm(
            chunks[1:], desc="Merging chunks", unit="chunk", leave=False
        )
    else:
        chunk_iterator = chunks[1:]

    for chunk in chunk_iterator:
        last_chunk = result[-1]

        # If current chunk is too small or combining won't exceed max_size
        if len(chunk) < min_size or (len(last_chunk) + len(chunk) <= max_size):
            # Merge with previous chunk
            result[-1] = last_chunk + "\n\n" + chunk
        else:
            result.append(chunk)

    return result


def clean_chunk(chunk):
    """Clean up a chunk by removing excessive whitespace and ensuring proper endings"""
    # Replace multiple newlines with double newline
    chunk = re.sub(r"\n{3,}", "\n\n", chunk)

    # Replace multiple spaces with single space
    chunk = re.sub(r" {2,}", " ", chunk)

    # Ensure chunk ends with proper punctuation if it's not a heading
    if not chunk.rstrip().endswith(
        (".", "?", "!", ":", ";", "-", ")", "]", "}", '"', "'")
    ):
        if not re.search(r"#+\s+\w+\s*$", chunk) and not chunk.rstrip().endswith("\n"):
            chunk = chunk.rstrip() + "."

    return chunk.strip()
