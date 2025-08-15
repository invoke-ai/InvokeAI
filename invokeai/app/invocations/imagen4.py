from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests
from PIL import Image
from io import BytesIO

from invokeai.app.invocations.baseinvocation import (
	BaseInvocation,
	Classification,
	invocation,
)
from invokeai.app.invocations.fields import (
	FieldDescriptions,
	ImageField,
	InputField,
	UIComponent,
)
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@dataclass
class _ImagenApiConfig:
	api_key: str
	model: str
	endpoint: str


def _get_api_key() -> Optional[str]:
	# Prefer common env vars
	return (
		os.environ.get("INVOKEAI_GEMINI_API_KEY")
		or os.environ.get("GEMINI_API_KEY")
		or os.environ.get("GOOGLE_API_KEY")
	)


def _build_config(model_override: Optional[str]) -> _ImagenApiConfig:
	api_key = _get_api_key()
	if not api_key:
		raise RuntimeError(
			"Google Gemini API key not found. Set INVOKEAI_GEMINI_API_KEY or GEMINI_API_KEY or GOOGLE_API_KEY."
		)
	model = model_override or "imagen-4.0-fast"
	endpoint = (
		f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
	)
	return _ImagenApiConfig(api_key=api_key, model=model, endpoint=endpoint)


def _extract_inline_images(candidate: dict[str, Any]) -> Iterable[bytes]:
	content = candidate.get("content") or {}
	parts = content.get("parts") or []
	for part in parts:
		inline = part.get("inline_data") or part.get("inlineData")
		if not inline:
			continue
		data = inline.get("data")
		if not data:
			continue
		try:
			yield base64.b64decode(data)
		except Exception:
			continue


def _parse_images_from_response(resp_json: dict[str, Any]) -> list[bytes]:
	images: list[bytes] = []
	candidates = resp_json.get("candidates") or []
	for cand in candidates:
		for img_bytes in _extract_inline_images(cand):
			images.append(img_bytes)
	return images


@invocation(
	"imagen4_fast",
	title="Imagen 4 Fast (Gemini)",
	tags=["imagen4", "gemini", "api"],
	category="image",
	version="1.0.0",
	classification=Classification.Prototype,
)
class Imagen4FastInvocation(BaseInvocation):
	"""Generate images with Google Imagen 4 Fast via the Gemini API.

	Requires an API key provided via environment variable `INVOKEAI_GEMINI_API_KEY`, `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
	"""

	prompt: str = InputField(
		description="Text prompt to generate the image.", ui_component=UIComponent.Textarea
	)
	model: Optional[str] = InputField(
		default=None, description="Override model id (default imagen-4.0-fast)."
	)
	count: int = InputField(default=1, ge=1, le=4, description="Number of images to generate (max 4).")
	mime_type: str = InputField(
		default="image/png", description="Desired output image MIME type (image/png or image/jpeg)."
	)

	def _request_payload(self) -> dict[str, Any]:
		payload: dict[str, Any] = {
			"contents": [
				{
					"role": "user",
					"parts": [{"text": self.prompt}],
				}
			],
			"generationConfig": {
				"response_mime_type": self.mime_type,
			},
		}
		# Some models honor "safety_settings"; we omit by default.
		return payload

	def invoke(self, context: InvocationContext) -> ImageOutput:
		cfg = _build_config(self.model)

		headers = {"Content-Type": "application/json"}
		payload = self._request_payload()

		# Attempt multiple calls if count > 1; many Gemini image models return one image per request.
		all_images: list[ImageField] = []
		for _ in range(self.count):
			resp = requests.post(cfg.endpoint, headers=headers, json=payload, timeout=60)
			if resp.status_code != 200:
				# Try to extract useful error message
				try:
					msg = resp.json().get("error", {}).get("message")
				except Exception:
					msg = resp.text
				raise RuntimeError(
					f"Imagen API request failed ({resp.status_code}): {msg}"
				)
			data = resp.json()
			images_b64 = _parse_images_from_response(data)
			if not images_b64:
				# Some responses place images directly at top-level 'inlineData'
				inline = data.get("inline_data") or data.get("inlineData")
				if inline and inline.get("data"):
					try:
						images_b64 = [base64.b64decode(inline["data"])]
					except Exception:
						images_b64 = []

			if not images_b64:
				raise RuntimeError("Imagen API returned no images. Check model name and prompt.")

			for img_bytes in images_b64:
				try:
					pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
				except Exception as e:
					raise RuntimeError(f"Failed to decode image from API: {e}")
				image_dto = context.images.save(image=pil_img)
				all_images.append(ImageField(image_name=image_dto.image_name))

		# Return the first image for node output; additional images are saved to the gallery
		primary = all_images[0]
		# Load it to get dimensions
		pil_first = context.images.get_pil(primary.image_name)
		return ImageOutput(image=primary, width=pil_first.width, height=pil_first.height)