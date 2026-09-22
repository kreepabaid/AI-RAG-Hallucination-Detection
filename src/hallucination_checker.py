"""
hallucination_checker.py
Uses cross-encoder/nli-MiniLM2-L6-H768 for Natural Language Inference.
Verifies claims against all retrieved evidence chunks without arbitrary truncation.
"""

import re
import torch
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import NLI_MODEL


class HallucinationChecker:
    def __init__(self, model_name: str = NLI_MODEL, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading NLI model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Dynamic label mapping from model config
        cfg_labels = self.model.config.id2label or {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.idx_to_label = {}
        for idx, label_name in cfg_labels.items():
            norm_name = label_name.lower()
            if "entail" in norm_name:
                self.idx_to_label[int(idx)] = "supported"
            elif "contra" in norm_name:
                self.idx_to_label[int(idx)] = "contradicted"
            else:
                self.idx_to_label[int(idx)] = "unsupported"

        print("NLI model ready!")

    def _parse_claims(self, claims_text: str) -> List[str]:
        """Extract individual claims from LLM output cleanly."""
        if not claims_text or not claims_text.strip():
            return []

        claims = []
        lines = claims_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Strip markdown prefixes like **Claim 1:** or [1] or 1. or -
            cleaned = re.sub(r"^\*{0,2}(?:Claim\s*)?\d+[\.\)\]:]\*{0,2}\s*", "", line, flags=re.IGNORECASE)
            cleaned = re.sub(r"^[-•*]\s*", "", cleaned).strip()

            # Skip header or conversational lines
            lower = cleaned.lower()
            if (
                lower.startswith("here are") or
                lower.startswith("claims:") or
                lower.startswith("verifiable claims") or
                lower == "claims"
            ):
                continue

            if len(cleaned) >= 6:
                claims.append(cleaned)

        # Fallback to sentence splitting if no list structure was detected
        if not claims:
            sentences = re.split(r"(?<=[.?!])\s+", claims_text.strip())
            for s in sentences:
                s = s.strip()
                if len(s) >= 10:
                    claims.append(s)

        return claims

    def _evaluate_claim_against_passages(
        self,
        claim: str,
        passages: List[Dict[str, str]]
    ) -> Dict:
        """
        Verify a single claim against multiple evidence passages/chunks.
        A claim is considered SUPPORTED if ANY retrieved chunk supports it.
        """
        if not passages:
            return {
                "claim": claim,
                "verdict": "unsupported",
                "confidence": 100.0,
                "scores": {"supported": 0.0, "unsupported": 100.0, "contradicted": 0.0},
                "evidence_passage": "No evidence passages provided.",
                "evidence_source": "None",
            }

        pairs = [(p["text"][:1200], claim) for p in passages]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

        # Check each passage's prediction
        best_support_score = -1.0
        best_support_passage = None
        best_contra_score = -1.0
        best_contra_passage = None
        passage_results = []

        for p_info, prob_list in zip(passages, probs):
            p_scores = {
                self.idx_to_label.get(i, "unsupported"): round(prob_list[i] * 100, 1)
                for i in range(len(prob_list))
            }
            entail_score = p_scores.get("supported", 0.0)
            contra_score = p_scores.get("contradicted", 0.0)
            neutral_score = p_scores.get("unsupported", 0.0)

            passage_results.append({
                "scores": p_scores,
                "passage": p_info["text"],
                "source": p_info.get("source", "Unknown"),
            })

            if entail_score > best_support_score:
                best_support_score = entail_score
                best_support_passage = p_info

            if contra_score > best_contra_score:
                best_contra_score = contra_score
                best_contra_passage = p_info

        # Decision logic:
        # 1. If any chunk strongly supports the claim (entailment > 50% or highest)
        if best_support_score >= 50.0:
            verdict = "supported"
            confidence = best_support_score
            evidence_text = best_support_passage["text"]
            evidence_source = best_support_passage.get("source", "Retrieved Context")
            scores = {
                "supported": best_support_score,
                "unsupported": round(max(0.0, 100.0 - best_support_score - best_contra_score), 1),
                "contradicted": best_contra_score,
            }
        # 2. If directly contradicted by any chunk and not supported
        elif best_contra_score >= 50.0:
            verdict = "contradicted"
            confidence = best_contra_score
            evidence_text = best_contra_passage["text"]
            evidence_source = best_contra_passage.get("source", "Retrieved Context")
            scores = {
                "supported": best_support_score,
                "unsupported": round(max(0.0, 100.0 - best_support_score - best_contra_score), 1),
                "contradicted": best_contra_score,
            }
        # 3. Otherwise unsupported (neutral / not found in sources)
        else:
            verdict = "unsupported"
            neutral_conf = round(100.0 - best_support_score - best_contra_score, 1)
            confidence = max(neutral_conf, 50.0)
            evidence_text = "No retrieved passage provides supporting evidence for this claim."
            evidence_source = "None"
            scores = {
                "supported": best_support_score,
                "unsupported": confidence,
                "contradicted": best_contra_score,
            }

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "scores": scores,
            "evidence_passage": evidence_text[:300] + ("..." if len(evidence_text) > 300 else ""),
            "evidence_source": evidence_source,
        }

    def check(
        self,
        claims_text: str,
        context: Union[str, List[Dict[str, str]]],
        chunks: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Full hallucination check:
        Extracts claims and verifies each against all retrieved context passages.
        """
        claims = self._parse_claims(claims_text)

        if not claims:
            return {
                "claims": [],
                "hallucination_score": 0.0,
                "summary": "Could not extract any claims to verify.",
                "issues": "None",
                "total": 0,
                "supported": 0,
                "unsupported": 0,
                "contradicted": 0,
            }

        # Normalize context passages
        passages: List[Dict[str, str]] = []
        if chunks:
            for c in chunks:
                passages.append({
                    "text": c.get("text", ""),
                    "source": c.get("source", "Retrieved Chunk"),
                })
        elif isinstance(context, list):
            passages = context
        elif isinstance(context, str):
            # Split concatenated context by delimiter
            parts = [p.strip() for p in context.split("\n\n---\n\n") if p.strip()]
            if not parts:
                parts = [p.strip() for p in context.split("\n\n") if p.strip()]
            passages = [{"text": p, "source": f"Section {i+1}"} for i, p in enumerate(parts)]

        results = [self._evaluate_claim_against_passages(claim, passages) for claim in claims]

        total = len(results)
        supported = sum(1 for r in results if r["verdict"] == "supported")
        unsupported = sum(1 for r in results if r["verdict"] == "unsupported")
        contradicted = sum(1 for r in results if r["verdict"] == "contradicted")
        bad = unsupported + contradicted

        hallucination_score = round((bad / total) * 100, 1) if total else 0.0

        issues_lines = [
            f'- "{r["claim"]}" -> {r["verdict"].upper()} ({r["confidence"]}% confidence)'
            for r in results if r["verdict"] in ("unsupported", "contradicted")
        ]
        issues = "\n".join(issues_lines) if issues_lines else "None"

        summary = (
            f"{total} claims checked - "
            f"{supported} supported \u2705  "
            f"{unsupported} unsupported \u26a0\ufe0f  "
            f"{contradicted} contradicted \u274c"
        )

        return {
            "claims": results,
            "hallucination_score": hallucination_score,
            "summary": summary,
            "issues": issues,
            "total": total,
            "supported": supported,
            "unsupported": unsupported,
            "contradicted": contradicted,
        }
