"""
LLM-based defect signal extraction engine.

Handles:
- Batch processing of complaint narratives
- API calls to local LM Studio (or OpenAI-compatible endpoint)
- JSON response parsing and validation
- Retry logic with exponential backoff
- Progress tracking and checkpointing
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from nhtsa_pipeline.schema import (
    DefectAnnotation,
    DefectCategory,
    Severity,
    Subsystem,
    build_batch_prompt,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMExtractionError(Exception):
    """Raised when LLM extraction fails."""
    pass


class LLMExtractor:
    """Extract structured defect annotations from complaint narratives using an LLM.

    Connects to LM Studio's OpenAI-compatible API (or any OpenAI-compatible endpoint).
    Processes narratives in batches for efficiency.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen/qwen3.6-35b-a3b",
        temperature: float = 0.1,
        timeout_seconds: int = 180,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    def _build_messages(self, narrative: str) -> list[dict]:
        """Build the messages array for the chat completion API."""
        prompt = f"""{SYSTEM_PROMPT}

CURRENT COMPLAINT:
Narrative: {narrative}

Output: Return ONLY the JSON object, no markdown, no explanation."""

        return [
            {"role": "system", "content": "You are an automotive safety analyst. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

    def extract_single(self, narrative: str) -> dict[str, Any] | None:
        """Extract defect annotation from a single narrative.

        Args:
            narrative: Free-text complaint narrative

        Returns:
            Dict with extracted fields, or None on failure
        """
        messages = self._build_messages(narrative)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "temperature": self.temperature,
                        "messages": messages,
                        "max_tokens": 1024,
                    },
                )
                response.raise_for_status()

                content = response.json()["choices"][0]["message"]["content"].strip()

                # Extract JSON from response (handle markdown fences)
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if not json_match:
                    logger.warning(f"No JSON found in response: {content[:200]}")
                    continue

                result = json.loads(json_match.group(0))

                # Validate severity range
                if "severity" in result:
                    result["severity"] = max(1, min(5, int(result["severity"])))

                # Validate confidence range
                if "confidence" in result:
                    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

                return result

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt})")
                    time.sleep(delay)
                else:
                    logger.error(f"HTTP error: {e}")
                    break
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Error (attempt {attempt}/{self.max_retries}): {e}, retrying in {delay}s")
                    time.sleep(delay)
                else:
                    logger.error(f"Extraction failed after {self.max_retries} attempts: {e}")
                    return None

        return None

    def extract_batch(
        self,
        narratives: list[str],
        batch_size: int = 1,
    ) -> list[dict[str, Any] | None]:
        """Extract defect annotations from a batch of narratives.

        Processes one at a time (LM Studio doesn't support batch chat completions
        with individual JSON outputs).

        Args:
            narratives: List of complaint narratives
            batch_size: Not used (processed sequentially for JSON validation)

        Returns:
            List of extraction results (one per narrative, None on failure)
        """
        results = []
        for i, narrative in enumerate(tqdm(narratives, desc="Extracting")):
            result = self.extract_single(narrative)
            results.append(result)

            # Progress logging
            if (i + 1) % 100 == 0:
                success = sum(1 for r in results if r is not None)
                logger.info(f"  Processed {i + 1}/{len(narratives)} ({success} successful)")

        return results

    def check_health(self) -> tuple[bool, str]:
        """Check if the LLM endpoint is reachable.

        Returns:
            (reachable, message)
        """
        try:
            response = self.client.get("/models")
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return True, f"Connected. Available models: {', '.join(models[:5])}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


def extract_annotations(
    df: pd.DataFrame,
    output_path: str | Path,
    base_url: str = "http://localhost:1234/v1",
    model: str = "qwen/qwen3.6-35b-a3b",
    batch_size: int = 1000,
    resume: bool = True,
) -> pd.DataFrame:
    """Extract defect annotations from a DataFrame of complaints.

    Args:
        df: DataFrame with at least 'Narrative' and 'Complaint ID' columns
        output_path: Path to save intermediate results (for resume capability)
        base_url: LM Studio API base URL
        model: Model name to use
        batch_size: Number of complaints to process per logging interval
        resume: If True, skip already-extracted complaints

    Returns:
        DataFrame with extracted annotations
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing results (resume capability)
    processed_ids = set()
    if resume and output_path.exists():
        existing = pd.read_parquet(output_path)
        processed_ids = set(existing["complaint_id"].unique())
        logger.info(f"Resuming: {len(processed_ids)} already extracted")

    # Filter to unprocessed complaints
    mask = ~df["Complaint ID"].isin(processed_ids) if processed_ids else pd.Series(True, index=df.index)
    to_process = df[mask].copy()

    if len(to_process) == 0:
        logger.info("All complaints already extracted. Loading existing results.")
        return pd.read_parquet(output_path)

    logger.info(f"Extracting {len(to_process):,} new annotations...")

    # Initialize extractor
    extractor = LLMExtractor(base_url=base_url, model=model)
    health_ok, health_msg = extractor.check_health()
    logger.info(f"LLM Health: {health_msg}")

    if not health_ok:
        raise LLMExtractionError(f"LLM endpoint not reachable: {health_msg}")

    try:
        # Process in batches
        all_results = []

        for start in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
            batch = to_process.iloc[start:start + batch_size]
            narratives = batch["Narrative"].tolist()
            complaint_ids = batch["Complaint ID"].tolist()
            makes = batch.get("Make", [""] * len(batch)).tolist()
            models = batch.get("Model", [""] * len(batch)).tolist()
            model_years = batch.get("Model Year", [0] * len(batch)).tolist()
            dates = batch.get("Date Received", [""] * len(batch)).tolist()

            # Extract
            extractions = extractor.extract_batch(narratives)

            # Build annotation records
            for i, (comp_id, extr) in enumerate(zip(complaint_ids, extractions)):
                if extr is None:
                    continue

                try:
                    annotation = DefectAnnotation(
                        complaint_id=comp_id,
                        make=makes[i] if i < len(makes) else "",
                        model=models[i] if i < len(models) else "",
                        model_year=int(model_years[i]) if i < len(model_years) else 0,
                        date_received=str(dates[i]) if i < len(dates) else "",
                        narrative=narratives[i],
                        defect_category=DefectCategory(extr.get("defect_category", "other")),
                        severity=Severity(extr.get("severity", 3)),
                        subsystem=Subsystem(extr.get("subsystem", "other")),
                        symptoms=extr.get("symptoms", []),
                        recurrence=extr.get("recurrence", False),
                        time_to_failure_months=extr.get("time_to_failure_months"),
                        confidence=extr.get("confidence", 1.0),
                        extraction_model=model,
                    )
                    all_results.append(annotation.to_dict())
                except (ValidationError, ValueError, KeyError) as e:
                    logger.debug(f"Validation error for {comp_id}: {e}")
                    continue

            # Save intermediate results periodically
            if (start // batch_size + 1) % 5 == 0 or start + batch_size >= len(to_process):
                if all_results:
                    result_df = pd.DataFrame(all_results)
                    result_df.to_parquet(output_path, index=False)
                    logger.info(f"  Checkpoint: {len(all_results):,} annotations saved to {output_path}")

        # Final save
        if all_results:
            result_df = pd.DataFrame(all_results)
            result_df.to_parquet(output_path, index=False)
            logger.info(f"Extraction complete: {len(all_results):,} annotations saved")
        else:
            logger.warning("No annotations were extracted")

        return pd.DataFrame(all_results) if all_results else pd.DataFrame()

    finally:
        extractor.close()
