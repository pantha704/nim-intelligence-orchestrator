import re
from dataclasses import dataclass, field


@dataclass
class Candidate:
    name: str
    model: str
    content: str
    reasoning: str = ""
    latency_ms: float = 0
    error: str = ""


def normalize_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        answer = sentences[0]
    else:
        answer = text

    core = re.sub(r"[^\w\s]", "", answer.lower())
    core = re.sub(r"\s+", " ", core).strip()

    if len(core) > 80:
        if "|" in text:
            pipe_parts = [p.strip() for p in text.split("|")]
            return pipe_parts[0].lower()[:80]

        if "=" in text and "==" not in text:
            eq_parts = [p.strip() for p in text.split("=")]
            return eq_parts[0].lower()[:80]

        core = core[:80]

    return core


@dataclass
class ClusteringResult:
    clusters: list[list[Candidate]] = field(default_factory=list)
    disagreement_level: str = "none"
    leader: Candidate | None = None

    @property
    def is_agreement(self) -> bool:
        return self.disagreement_level == "none" and len(self.clusters) <= 1 or (
            self.disagreement_level != "high"
            and len(self.clusters) > 0
        )


def cluster_candidates(candidates: list[Candidate]) -> ClusteringResult:
    valid = [c for c in candidates if c.error == "" and c.content]
    if len(valid) == 0:
        return ClusteringResult()
    if len(valid) == 1:
        return ClusteringResult(
            clusters=[[valid[0]]],
            disagreement_level="none",
            leader=valid[0],
        )

    clusters: list[list[Candidate]] = []
    for candidate in valid:
        normalized = normalize_answer(candidate.content)
        matched = False
        for cluster in clusters:
            cluster_normalized = normalize_answer(cluster[0].content)
            if _answers_match(normalized, cluster_normalized):
                cluster.append(candidate)
                matched = True
                break
        if not matched:
            clusters.append([candidate])

    leader = max(clusters, key=lambda c: len(c))[0]

    num_clusters = len(clusters)
    if num_clusters == 1:
        disagreement_level = "none"
    elif num_clusters == 2 and len(clusters[1]) == 1 and len(clusters[0]) >= len(valid) - 1:
        disagreement_level = "low"
    else:
        disagreement_level = "high"

    return ClusteringResult(
        clusters=clusters,
        disagreement_level=disagreement_level,
        leader=leader,
    )


def _answers_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True

    nums_a = set(re.findall(r"\d+", a))
    nums_b = set(re.findall(r"\d+", b))
    if nums_a or nums_b:
        if nums_a != nums_b:
            return False
        if nums_a == nums_b and nums_a:
            return True

    if " " not in a and " " not in b and (a in b or b in a):
        return True
    words_a = a.split()
    words_b = b.split()
    if min(len(words_a), len(words_b)) == 0:
        return False
    overlap = sum(1 for w in words_a if w in words_b)
    min_len = min(len(words_a), len(words_b))
    if min_len > 0:
        overlap_ratio = overlap / min_len
        if min_len <= 3 and overlap_ratio < 1.0:
            return False
        if overlap_ratio >= 0.7:
            return True

    return False
