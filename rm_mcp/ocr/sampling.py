"""
Sampling-based OCR for reMarkable documents.

This module provides OCR functionality using MCP's sampling capability,
allowing the host application's LLM to extract text from images.

## Usage

Sampling OCR is only available when:
1. REMARKABLE_OCR_BACKEND is explicitly set to "sampling"
2. The client supports the sampling capability

The key advantage of sampling-based OCR is that it uses the client's own model,
which may provide better results for handwriting without requiring additional
API keys or services.

## Important Notes

- Sampling is asynchronous and requires a Context object from tool execution
- The prompt is carefully crafted to return ONLY the extracted text
- Returns None if sampling is not available or fails
"""

import asyncio
import base64
import io
from typing import TYPE_CHECKING, List, Optional

from mcp.types import ImageContent, ModelHint, ModelPreferences, SamplingMessage, TextContent

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


# Model preferences for OCR tasks - prioritize speed for sub-10s latency
# Haiku-class models handle handwriting at ~1-4s vs Opus at 15-45s (~90% accuracy)
OCR_MODEL_PREFERENCES = ModelPreferences(
    hints=[
        # Fast tier — Haiku-class: 1-4s, adequate for handwriting OCR
        ModelHint(name="claude-haiku-4-5"),
        ModelHint(name="claude-haiku-4"),
        ModelHint(name="claude-haiku-3-5"),
        ModelHint(name="gemini-flash"),
        ModelHint(name="gpt-4o-mini"),
        # Mid tier — Sonnet: 5-12s, better on complex handwriting
        ModelHint(name="claude-sonnet-4.5"),
        ModelHint(name="claude-sonnet-4"),
        ModelHint(name="claude-3-5-sonnet"),
        ModelHint(name="gemini-1.5-flash"),
        # Fallback tier — Opus only if nothing faster is available
        ModelHint(name="claude-opus-4.5"),
        ModelHint(name="claude-opus-4"),
        ModelHint(name="gpt-4o"),
        ModelHint(name="gemini-2.5-pro"),
    ],
    intelligencePriority=0.3,  # Speed matters more than max intelligence for OCR
    speedPriority=0.9,  # Prioritize fast models
    costPriority=0.0,
)

# Max dimension for the longest side of PNG before sending to OCR model.
# reMarkable pages are up to 1872px tall. Halving to 936px reduces base64
# payload by ~4x and speeds up model inference with minimal accuracy loss.
_OCR_MAX_IMAGE_DIMENSION = int(__import__("os").environ.get("REMARKABLE_OCR_MAX_DIMENSION", "936"))


def _resize_for_ocr(png_data: bytes) -> bytes:
    """Resize PNG to _OCR_MAX_IMAGE_DIMENSION on the longest side before OCR.

    Reduces upload size and model inference time with minimal accuracy loss
    for handwriting recognition on reMarkable pages.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png_data))
        w, h = img.size
        if max(w, h) <= _OCR_MAX_IMAGE_DIMENSION:
            return png_data

        scale = _OCR_MAX_IMAGE_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        return png_data  # Fall back to original on any error


# The OCR prompt is carefully designed to extract ONLY the text content
# with no additional commentary, explanations, or formatting.
OCR_SYSTEM_PROMPT = """You are an OCR system. Extract the exact text visible in the image.

CRITICAL RULES:
1. Output ONLY the text found in the image, nothing else
2. Do NOT add any commentary, explanations, or descriptions
3. Do NOT use phrases like "The text says:" or "I can see:"
4. Do NOT describe the image or its contents
5. Preserve the original text layout and line breaks where possible
6. If no text is visible, output exactly: [NO TEXT DETECTED]
7. If text is unclear, transcribe what you can and use [...] for unclear portions

You are extracting handwritten notes from a reMarkable tablet. Focus on accuracy."""

OCR_USER_PROMPT = "Extract all text from this image. Output only the text content, nothing else."


async def ocr_via_sampling(
    ctx: "Context",
    png_data: bytes,
    max_tokens: int = 2000,
) -> Optional[str]:
    """
    Perform OCR on an image using the client's LLM via MCP sampling.

    Args:
        ctx: The FastMCP Context object from a tool function
        png_data: PNG image bytes to perform OCR on
        max_tokens: Maximum tokens for the response (default: 2000)

    Returns:
        Extracted text from the image, or None if OCR failed

    Example:
        @mcp.tool()
        async def my_ocr_tool(document: str, ctx: Context) -> str:
            # ... get png_data from document ...
            text = await ocr_via_sampling(ctx, png_data)
            if text:
                return text
            return "OCR failed"
    """
    try:
        session = ctx.session
        if not session:
            return None

        # Resize image before encoding — reduces upload size and inference time
        png_data = _resize_for_ocr(png_data)

        # Encode image as base64
        image_b64 = base64.b64encode(png_data).decode("utf-8")

        # Create the sampling messages with image and text prompt.
        # Note: MCP's SamplingMessage accepts a single content item (TextContent
        # or ImageContent), not a list of content items. This means we need two
        # consecutive "user" role messages — one for the image and one for the
        # text prompt. This is a limitation of the SamplingMessage API; if it
        # supported multi-part content, these would be combined into one message.
        messages = [
            SamplingMessage(
                role="user",
                content=ImageContent(
                    type="image",
                    data=image_b64,
                    mimeType="image/png",
                ),
            ),
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=OCR_USER_PROMPT),
            ),
        ]

        # Request completion from the client's LLM
        # Use model preferences to request a capable vision model
        result = await session.create_message(
            messages=messages,
            system_prompt=OCR_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.0,  # Use low temperature for consistency
            model_preferences=OCR_MODEL_PREFERENCES,
        )

        # Extract text from the result
        if result and result.content:
            if isinstance(result.content, TextContent):
                text = result.content.text
            elif hasattr(result.content, "text"):
                text = result.content.text
            else:
                return None

            # Check for "no text" response
            if text and "[NO TEXT DETECTED]" not in text:
                return text.strip()

        return None

    except Exception:
        # Sampling may fail for various reasons: client doesn't support sampling,
        # session is not available, model doesn't support vision, network issues, etc.
        # We intentionally swallow all exceptions and return None.
        return None


async def ocr_pages_via_sampling(
    ctx: "Context",
    png_data_list: List[bytes],
    max_tokens: int = 2000,
) -> Optional[List[str]]:
    """
    Perform OCR on multiple pages using the client's LLM via MCP sampling.

    Args:
        ctx: The FastMCP Context object from a tool function
        png_data_list: List of PNG image bytes to perform OCR on
        max_tokens: Maximum tokens for each response (default: 2000)

    Returns:
        List of extracted text (one per page), or None if all pages failed
    """

    async def _ocr_one(png_data: bytes) -> str:
        if not png_data:
            return ""
        text = await ocr_via_sampling(ctx, png_data, max_tokens)
        return text or ""

    # Run all pages concurrently — total time ≈ slowest single page
    results = list(await asyncio.gather(*[_ocr_one(p) for p in png_data_list]))
    return results if any(results) else None


def get_ocr_backend() -> str:
    """
    Get the configured OCR backend from the environment.

    Returns the raw value from REMARKABLE_OCR_BACKEND env var (default: "sampling").
    The only supported value is "sampling".

    Sampling OCR requires a client that supports the sampling capability.
    """
    import os

    return os.environ.get("REMARKABLE_OCR_BACKEND", "sampling").lower()


def should_use_sampling_ocr(ctx: "Context") -> bool:
    """
    Check if sampling-based OCR should be used.

    Returns True if:
    1. REMARKABLE_OCR_BACKEND is explicitly set to "sampling", AND
    2. The client supports the sampling capability

    Args:
        ctx: The FastMCP Context object

    Returns:
        True if sampling OCR should be used, False otherwise
    """
    from rm_mcp.capabilities import client_supports_sampling

    backend = get_ocr_backend()

    # Only use sampling if explicitly configured
    if backend != "sampling":
        return False

    # Check if client supports sampling
    return client_supports_sampling(ctx)
