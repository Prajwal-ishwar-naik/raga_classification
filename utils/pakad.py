from collections import Counter


def detect_pakads(swara_sequence, top_k=10):
    if not swara_sequence or len(swara_sequence) < 4:
        return []

    # 1. Remove consecutive duplicates
    cleaned = [swara_sequence[0]]
    for s in swara_sequence[1:]:
        if s != cleaned[-1]:
            cleaned.append(s)

    # 2. Normalize
    cleaned = [s.lower() for s in cleaned]

    phrases = []

    # 3. Use meaningful window sizes
    for length in [4, 5]:
        for i in range(len(cleaned) - length + 1):
            window = cleaned[i : i + length]

            # Remove weak patterns
            if len(set(window)) <= 2:
                continue

            phrases.append("-".join(window))

    counts = Counter(phrases)
    return counts.most_common(top_k)
