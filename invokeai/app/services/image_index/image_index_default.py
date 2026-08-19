import os
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from invokeai.app.services.image_index.image_index_base import ImageIndexServiceBase, TextSearchUnavailableError
from invokeai.app.services.image_index.image_index_common import EMBEDDING_DTYPE, ImageIndexStatus
from invokeai.app.services.image_index.projection import compute_umap, projection_params, scope_hash
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.session_queue.session_queue_common import DEFAULT_QUEUE_ID
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.util.devices import TorchDevice

if TYPE_CHECKING:
    from logging import Logger

    from invokeai.app.services.config.config_default import InvokeAIAppConfig
    from invokeai.app.services.invoker import Invoker
    from invokeai.backend.model_manager.configs.factory import AnyModelConfig

# A batch of PIL images in, an (N, dim) float32 embedding matrix out.
EncodeFn = Callable[[list[Image.Image]], np.ndarray]


def _is_cpu_override(override: Optional[str]) -> bool:
    """True if `image_index_device` asks for CPU mode.

    `cpu` is the only meaningful value; the config documents every other value as ignored.
    Compared as a normalized string rather than via `torch.device`, which raises on anything
    that is not a valid device spec — including near-misses like `CPU`.
    """
    return override is not None and override.strip().lower() == "cpu"


# How long the worker sleeps between generation-idle checks and empty-queue polls.
_POLL_SECONDS = 1.0

# An image is retried this many times before being marked permanently failed. This bounds
# per-image badness only — a corrupt file, an unreadable image, an embedding that cannot be
# normalized. Failures of the machinery (the encoder, the database) never consume this budget;
# see _record_systemic_failure.
_MAX_ATTEMPTS = 3

# Consecutive systemic failures back off exponentially from _POLL_SECONDS up to this ceiling,
# so an outage that lasts hours does not retry at 1 Hz while still recovering promptly.
_MAX_BACKOFF_SECONDS = 60.0


def _normalize_query_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize one query embedding, refusing anything that would poison a search.

    The norm is taken in float64 for the same reason `_process_batch` does it: a float32 sum of
    squares under/overflows well inside the range these encoders produce.

    A zero or non-finite norm is raised rather than passed through. The indexer can afford to
    drop such a row; a query cannot. Dividing by a non-finite norm gives a NaN vector, every
    similarity it takes part in is NaN, `argpartition` then returns arbitrary rows, and FastAPI
    serializes the NaN scores as bare `NaN` — which is not valid JSON, so the browser fails to
    parse the response instead of showing an empty result.
    """
    norm = float(np.linalg.norm(vector.astype(np.float64)))

    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("The embedding model returned a degenerate vector for this query")

    return (vector / norm).astype(EMBEDDING_DTYPE)


# Two phrases that share no content words. Any working text tower puts them well
# apart (~0.6 cosine for CLIP); a broken one maps them to the same vector.
_TEXT_PROBE_PAIR = ("a photograph of a cat", "an engineering diagram")
# Not 1.0: the failure is exact degeneracy, but float paths and fp16 towers can
# land a hair below it, and no usable encoder comes anywhere near this.
_TEXT_PROBE_MAX_COSINE = 0.999


def _text_encoder_defect(tokenizer: Any, model: Any, is_siglip: bool) -> Optional[str]:
    """Why this text encoder is unusable, or None if it tells two phrases apart.

    A model directory with no tokenizer files does not fail to load: transformers
    resolves the tokenizer class from config.json and builds it with an empty
    vocabulary, which maps every word to the same unknown token. The tower then
    faithfully encodes that, so every phrase - every vocabulary label, every
    search query - comes back as the same vector. Nothing downstream can detect
    it: cluster labelling still returns a label for every cluster, chosen by
    float noise among identical scores, and text search still returns a ranking.
    One forward pass at load time is what separates that from a working encoder.
    """
    try:
        with torch.no_grad():
            # Same padding rule as _embed_texts: SigLIP was trained pad-to-max-length.
            inputs = tokenizer(
                list(_TEXT_PROBE_PAIR),
                padding="max_length" if is_siglip else True,
                return_tensors="pt",
                truncation=True,
            )
            outputs = model(**inputs)
            embeddings = outputs.pooler_output if is_siglip else outputs.text_embeds
            matrix = embeddings.float().cpu().numpy()
    except Exception as e:
        # A probe that cannot complete has not cleared the encoder, and the
        # shapes it dies on are the same broken installs it exists to catch: a
        # tokenizer whose ids overrun the tower's vocabulary raises IndexError
        # inside nn.Embedding, and one that loaded without a model_max_length
        # raises ValueError on the SigLIP pad-to-max-length path. Reported as a
        # defect rather than left to escape, which would reach the router untyped
        # — a 500 instead of the 409 this path exists to produce — and, being
        # remembered by nothing, reload the tower on the next keystroke.
        return f"could not be exercised ({type(e).__name__}: {e})"

    if not np.isfinite(matrix).all():
        return "produced a non-finite embedding"

    # float64 for the same reason _normalize_query_vector uses it, and here it is
    # load-bearing rather than defensive: in float32 a pair of *identical* rows
    # big enough to overflow the sum of squares gives infinite norms and a NaN
    # cosine, and `NaN >= threshold` is False — so the exact degeneracy this
    # function exists to catch was waved through as healthy. Every float32 value
    # is finite here (checked above), so widening puts both the norms and the dot
    # product out of overflow and underflow range for good.
    matrix = matrix.astype(np.float64)

    norms = np.linalg.norm(matrix, axis=1)
    if float(norms.min()) == 0.0:
        # Checked before the cosine, which would divide by zero here — and an
        # all-zero pair would otherwise score 0.0 and pass as "discriminating".
        return "produced an all-zero embedding"

    cosine = float(matrix[0] @ matrix[1] / (norms[0] * norms[1]))
    if cosine >= _TEXT_PROBE_MAX_COSINE:
        return f"maps unrelated phrases to the same embedding (cosine {cosine:.3f})"

    return None


def _normalize_batch(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row, zeroing the ones that cannot be normalized.

    Shares `_normalize_query_vector`'s float64 norm — a float32 sum of squares
    under/overflows inside the range these encoders produce — but not its
    refusal. A single query is worth failing loudly; a batch is not. The
    vocabulary build sends ~11,700 phrases through here, and aborting all of
    them over one degenerate row would discard minutes of encoder work, raise
    past the caller's error handling as a 500, and (nothing having been cached)
    do it all again on the next request.

    A zeroed row scores 0 against everything, so the phrase simply never wins a
    label — which is what a meaningless embedding should do.
    """
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    usable = np.isfinite(norms) & (norms > 0)

    return np.where(usable, matrix / np.where(usable, norms, 1.0), 0.0).astype(EMBEDDING_DTYPE)


def warm_up_attention(config: "InvokeAIAppConfig", logger: "Logger") -> None:
    """Run one tiny scaled_dot_product_attention call while the process is still single-threaded.

    ROCm's AOTriton backend initializes its SDPA kernels lazily on the first
    call, and that init is not thread-safe: two threads making their first
    call concurrently can get a spurious "Accelerated SDPA only supports..."
    error or crash the process. The image index adds the first threads that
    run attention concurrently with the session processor, so its startup
    wiring calls this before any service thread exists (a warmup issued any
    later — e.g. from the service's own start() — could itself race a
    generation resumed from a queue that was non-empty at shutdown).
    """
    if not config.image_index_enabled:
        return
    try:
        # Mirrors ImageIndexService._cpu_mode: `cpu` selects CPU mode, anything else is
        # ignored rather than parsed as a device, so a bad value cannot decide where — or
        # whether — the warm-up runs.
        if _is_cpu_override(config.image_index_device):
            return
        device = TorchDevice.choose_torch_device()
        if device.type != "cuda":
            return
        query = torch.zeros((1, 1, 8, 8), device=device, dtype=torch.float16)
        torch.nn.functional.scaled_dot_product_attention(query, query, query)
        torch.cuda.synchronize(device)
    except Exception:
        logger.warning("Attention warm-up failed", exc_info=True)


class ImageIndexService(ImageIndexServiceBase):
    """Embeds gallery images on a daemon worker thread.

    The image-service callbacks only enqueue work and set a status-dirty
    flag — they fire synchronously on the caller's thread and must never do
    I/O. All embedding and all status emission happens on the worker, which
    pauses while a generation is in progress unless it is configured to run
    on the CPU.
    """

    def __init__(self, encode_fn: Optional[EncodeFn] = None, model_id: Optional[str] = None) -> None:
        """Args:
        encode_fn: Test seam. When provided, it is used instead of loading the
            configured model, and `model_id` identifies the index it writes.
        model_id: Required when `encode_fn` is provided.
        """
        if (encode_fn is None) != (model_id is None):
            raise ValueError("encode_fn and model_id must be provided together")
        self._encode_fn_override = encode_fn
        self._model_id_override = model_id

        self._queue: Queue[str] = Queue()
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        # Per-image failure counts; after _MAX_ATTEMPTS the name moves to
        # _failed, which is excluded from backfill so a bad file cannot make
        # the backfill loop spin forever. Transient failures get retried.
        self._attempts: dict[str, int] = {}
        self._failed: set[str] = set()
        # Consecutive failures of the machinery rather than of any image. Drives the retry
        # backoff and is reset by the first batch that stores anything. Worker-thread only.
        self._systemic_failures = 0
        self._stop_event = threading.Event()
        self._backfill_pending = threading.Event()
        # Set by the image-service callbacks when counts changed; the worker
        # (the only emitter) turns it into one status event per sweep.
        self._status_dirty = threading.Event()
        # Owners of freshly embedded images, awaiting a per-user poke once
        # the index goes quiescent. Worker-thread only — no locking.
        self._pending_pokes: set[str] = set()
        self._worker: Optional[threading.Thread] = None
        # user_id -> all_images. Deduplicates projection requests per user;
        # a request arriving while that user's projection is being computed
        # lands here again and is honored on the next pass.
        self._projection_requests: dict[str, bool] = {}
        # user_id -> the scope hash whose failed fit has already been retried
        # this process; see _failed_scope_retry_available.
        self._failed_projection_scopes: dict[str, str] = {}
        self._projection_lock = threading.Lock()

        self._invoker: Optional["Invoker"] = None
        self._model_config: Optional["AnyModelConfig"] = None
        self._model_id: Optional[str] = None
        self._encode_fn: Optional[EncodeFn] = None
        # RAM-resident model used only in CPU mode; see _encode_with_model.
        self._cpu_model: Optional[Any] = None
        self._processor: Optional[Any] = None
        # Lazily-loaded (tokenizer, text_model, is_siglip); None until first text search.
        self._text_encoder: Optional[tuple[Any, Any, bool]] = None
        self._text_encoder_lock = threading.Lock()
        # Why the load-time probe rejected this text encoder. Reloading cannot
        # change the verdict — the invoker owns one service instance per run and
        # a new one is built from scratch, so the same directory is probed every
        # time — and each attempt pulls the whole text
        # tower (hundreds of MB) through MODEL_LOAD_LOCK, contending with
        # generation's model loads, only to reject it again. Without this that
        # happens once per search keystroke, because the encoder is never assigned
        # and so nothing downstream of the probe is cached. _vocab_failure does
        # the same job for cluster labels, which is the only path it covers.
        #
        # The message rather than the exception: a stored exception grows two
        # traceback frames per re-raise, forever, and the frames it captured at
        # the rejection keep the rejected tower itself alive.
        self._text_encoder_failure: Optional[str] = None
        # A vocabulary build that already failed. Without this, the very common
        # vision-only install retries AutoTokenizer.from_pretrained — inside
        # MODEL_LOAD_LOCK, contending with generation's model loads — once per
        # points refresh, forever.
        self._vocab_failure: Optional[Exception] = None
        # Set by a labels request, serviced on the worker: the build is minutes
        # of encoder work and must never run on a request thread.
        self._vocab_build_requested = threading.Event()
        # Guards the lazy _processor/_cpu_model init: embed_image runs on
        # request threads concurrently with the indexer worker.
        self._vision_init_lock = threading.Lock()
        # scope_hash -> (names, matrix): the accessible embedding matrix for
        # similarity search, so repeated queries skip re-reading BLOBs.
        self._search_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._search_cache_lock = threading.Lock()
        # (vocabulary, embeddings) for cluster labeling; RAM + disk cached.
        self._vocab_cache: Optional[tuple[list[str], np.ndarray]] = None
        self._vocab_lock = threading.Lock()

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def get_status(self) -> ImageIndexStatus | None:
        if self._invoker is None or self._model_id is None:
            return None
        status = self._invoker.services.image_index_records.count_index_status(self._model_id)
        return status.model_copy(update={"failed": len(self._failed)})

    def embed_image(self, image: Image.Image) -> np.ndarray:
        if self._encode_fn is None:
            raise RuntimeError("The image index is not running")
        rgb = image.convert("RGB")
        try:
            matrix = np.asarray(self._encode_fn([rgb]), dtype=EMBEDDING_DTYPE)
        except Exception:
            # The model cache deletes an entry whose load failed precisely so
            # the next attempt rebuilds it from disk (e.g. an entry left in a
            # bad state by VRAM contention with a concurrent generation).
            # One retry is the cache's designed recovery; the worker gets the
            # same effect from its batch retry loop.
            if self._invoker is not None:
                self._invoker.services.logger.warning(
                    "Image embed failed; retrying with a fresh model load", exc_info=True
                )
            matrix = np.asarray(self._encode_fn([rgb]), dtype=EMBEDDING_DTYPE)
        if matrix.ndim != 2 or matrix.shape[0] != 1:
            raise RuntimeError(f"Encoder returned shape {matrix.shape}; expected (1, D)")
        return _normalize_query_vector(matrix[0])

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batched text embedding, (N, D) L2-normalized."""
        if self._model_id is None:
            raise TextSearchUnavailableError("The image index is not running")

        tokenizer, model, is_siglip = self._get_text_encoder()
        with torch.no_grad():
            # SigLIP was trained with pad-to-max-length; pad-to-longest degrades
            # its text embeddings. CLIP uses ordinary longest-padding.
            inputs = tokenizer(texts, padding="max_length" if is_siglip else True, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            embeddings = outputs.pooler_output if is_siglip else outputs.text_embeds
            matrix = embeddings.float().cpu().numpy()
        return _normalize_batch(matrix)

    def get_vocab_embeddings(self) -> tuple[list[str], np.ndarray]:
        """The phrase-embedding matrix for cluster labelling.

        Never builds it: a first run embeds ~1700 phrases x 7 templates through
        a CPU text tower, which takes minutes. Callers run on the event loop's
        shared executor, so blocking there — worse, blocking there while holding
        `_vocab_lock` — starves every other `asyncio.to_thread` in the app,
        `/points` and image search included. The build is handed to the index
        worker instead and this raises until it lands.
        """
        with self._vocab_lock:
            if self._vocab_cache is not None:
                return self._vocab_cache
            if self._vocab_failure is not None:
                # Cleared again on the way out, not just when it was stored:
                # re-raising a persistent exception re-attaches a fresh traceback
                # to it, so without this the frames grow by one propagation per
                # labels request, forever, each set holding that request's caller
                # frame — whose `record` is the whole projection, coordinates
                # included. Clearing here bounds it to the one propagation in
                # flight instead, which the next request replaces.
                raise self._vocab_failure.with_traceback(None)
            if self._invoker is None or self._model_id is None:
                raise TextSearchUnavailableError("The image index is not running")

            self._vocab_build_requested.set()

        raise TextSearchUnavailableError("Cluster labels are still being prepared; try again shortly")

    def _build_vocab_embeddings(self) -> None:
        """Build and cache the phrase embeddings. Index-worker thread only."""
        from invokeai.app.services.image_index.cluster_labels import (
            ensemble_phrase_embeddings,
            load_vocabulary,
            vocab_fingerprint,
        )

        with self._vocab_lock:
            if self._vocab_cache is not None or self._vocab_failure is not None:
                return
            if self._invoker is None or self._model_id is None:
                return

            vocabulary = load_vocabulary()
            fingerprint = vocab_fingerprint(vocabulary)
            # The model hash contains ':' (e.g. 'blake3:...'), illegal in
            # Windows filenames.
            model_tag = self._model_id.replace(":", "_")[:24]
            cache_path = self._invoker.services.configuration.db_path.parent / f"cluster_vocab_{model_tag}.npz"

            if cache_path.exists():
                try:
                    # Closed explicitly: np.load returns a lazily-read NpzFile
                    # holding the zip handle open, and the rewrite below
                    # truncates this very file on a fingerprint mismatch.
                    with np.load(cache_path, allow_pickle=False) as cached:
                        if str(cached["fingerprint"]) == fingerprint:
                            self._vocab_cache = (vocabulary, cached["embeddings"].astype(EMBEDDING_DTYPE))

                    if self._vocab_cache is not None:
                        return
                except Exception:
                    self._invoker.services.logger.warning("Discarding unreadable cluster vocabulary cache")

            # First run for this model: embedding ~1700 phrases x 7 templates
            # takes minutes — hence the disk cache.
            try:
                embeddings = ensemble_phrase_embeddings(self._embed_texts, vocabulary)
            except Exception as e:
                self._invoker.services.logger.warning("Could not build the cluster vocabulary", exc_info=True)
                # Remembered, so a vision-only install answers the next request
                # from memory instead of reloading the absent text tower.
                #
                # Stripped of every frame it captured, and logged first because
                # stripping also empties the `exc_info=True` above. Those frames
                # reach the locals of whatever raised: since the text encoder is
                # now probed after loading rather than failing during it, a
                # rejection is raised with the loaded tower still bound, and
                # keeping the frames would keep hundreds of MB alive for the life
                # of the process. __cause__ is dropped for the same reason and is
                # not information: a load failure already names the underlying
                # error in its own message. The full detail is in the log above;
                # only type and message are ever re-raised.
                e.__cause__ = None
                e.__context__ = None
                self._vocab_failure = e.with_traceback(None)

                return

            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Written aside and renamed: two processes sharing a db_dir can
                # first-run at once, and a kill mid-write would otherwise leave
                # a truncated archive that costs another full re-embed.
                #
                # The staging name has to end in `.npz` because np.savez appends
                # that extension to any path that lacks it: written as `.tmp`,
                # the archive landed at `.tmp.npz` and the rename below then
                # failed on the `.tmp` that was never created. The failure was
                # swallowed by the handler, so the cache never reached disk and
                # every restart re-embedded the whole vocabulary (minutes,
                # during which cluster labels are unavailable) while leaking one
                # orphaned staging file per run.
                staging_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp.npz")
                np.savez(staging_path, embeddings=embeddings, fingerprint=np.str_(fingerprint))
                os.replace(staging_path, cache_path)
            except Exception:
                # With the exception, not just the path. The in-memory cache
                # below is assigned either way, so a write failure costs
                # nothing until the next restart and is invisible until someone
                # goes looking at startup times. The `.tmp` bug above survived
                # because this line said only that something had gone wrong,
                # never what — the FileNotFoundError it swallowed names the
                # missing staging file outright.
                self._invoker.services.logger.warning(
                    f"Could not write cluster vocabulary cache to {cache_path}", exc_info=True
                )
            self._vocab_cache = (vocabulary, embeddings)

    def _get_text_encoder(self) -> tuple[Any, Any, bool]:
        with self._text_encoder_lock:
            if self._text_encoder_failure is not None:
                raise TextSearchUnavailableError(self._text_encoder_failure)
            if self._text_encoder is None:
                if self._model_config is None:
                    raise TextSearchUnavailableError("The image index has no embedding model loaded")

                from transformers import AutoTokenizer, CLIPTextModelWithProjection, SiglipTextModel

                model_path = str(self._model_abs_path())
                is_siglip = self._model_config.type is ModelType.SigLIP
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                    text_cls = SiglipTextModel if is_siglip else CLIPTextModelWithProjection
                    # skip_torch_weight_init serializes nothing: it monkey-patches
                    # torch.nn.Linear/_ConvNd/Embedding.reset_parameters process-wide
                    # and restores whatever it saw on entry. Two threads inside it at
                    # once means the second saves the _no_op, and whoever leaves last
                    # restores _no_op for the life of the process — every layer built
                    # afterwards silently skips weight init. MODEL_LOAD_LOCK is what
                    # makes it safe, and also excludes the concurrent VRAM move whose
                    # load_state_dict(assign=True) would hijack these parameters onto
                    # the meta device. Same reasoning as ModelLoader._load_and_cache.
                    with MODEL_LOAD_LOCK.write_lock(), skip_torch_weight_init():
                        model = text_cls.from_pretrained(model_path, local_files_only=True)
                except Exception as e:
                    # Vision-only or partial installs (e.g. the IP-Adapter image
                    # encoder, or a full-CLIP checkpoint shipped without tokenizer
                    # files) fail in assorted ways — transformers raises OSError,
                    # TypeError, or ValueError depending on which file is missing.
                    # Image-similarity search still works either way.
                    raise TextSearchUnavailableError(
                        "The configured embedding model's text encoder could not be loaded "
                        f"({type(e).__name__}: {e}); install the full CLIP model, including its "
                        "tokenizer files, to enable semantic text search"
                    ) from e
                model.eval()
                defect = _text_encoder_defect(tokenizer, model, is_siglip)
                if defect is not None:
                    # Remembered, not just raised: this costs one load and one
                    # probe per process rather than one per request. See
                    # _text_encoder_failure.
                    self._text_encoder_failure = (
                        f"The configured embedding model's text encoder {defect}, so cluster labels and "
                        "text search would be meaningless. The usual cause is a model directory with no "
                        "tokenizer files (vocab.json, merges.txt): reinstall the embedding model to get "
                        "them, or configure image_index_model to a model that has them, then restart"
                    )
                    raise TextSearchUnavailableError(self._text_encoder_failure)
                self._text_encoder = (tokenizer, model, is_siglip)
            return self._text_encoder

    def get_accessible_embeddings(self, user_id: str | None) -> tuple[list[str], np.ndarray]:
        if self._invoker is None or self._model_id is None:
            return [], np.empty((0, 0), dtype=EMBEDDING_DTYPE)
        records = self._invoker.services.image_index_records

        names = records.list_accessible_embedded_images(user_id, self._model_id)
        if not names:
            return [], np.empty((0, 0), dtype=EMBEDDING_DTYPE)

        # Memory math: one entry is N x dim float32 (~300 MB at 100k x 768),
        # so the LRU holds two entries and evicts oldest-first rather than
        # clearing wholesale. Keys self-invalidate: any change to the
        # accessible set changes the scope hash.
        cache_key = scope_hash(self._model_id, names)
        with self._search_cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                # LRU touch.
                del self._search_cache[cache_key]
                self._search_cache[cache_key] = cached
        if cached is None:
            # Built outside the lock: a concurrent miss for the same scope
            # transiently builds a duplicate, which beats serializing every
            # search behind a multi-second BLOB read.
            cached = records.get_embeddings(names, self._model_id)
            with self._search_cache_lock:
                while len(self._search_cache) >= 2:
                    self._search_cache.pop(next(iter(self._search_cache)))
                self._search_cache[cache_key] = cached

        return cached

    def search_similar(self, user_id: str | None, query_embedding: np.ndarray, limit: int) -> list[tuple[str, float]]:
        found_names, matrix = self.get_accessible_embeddings(user_id)
        if matrix.size == 0:
            return []

        scores = matrix @ query_embedding.astype(matrix.dtype)
        top = min(limit, len(found_names))
        order = np.argpartition(-scores, top - 1)[:top]
        order = order[np.argsort(-scores[order])]
        return [(found_names[index], float(scores[index])) for index in order]

    def request_projection(
        self,
        user_id: str,
        all_images: bool = False,
        failed_scope: Optional[str] = None,
        user_initiated: bool = False,
    ) -> bool:
        if self._model_id is None or self._worker is None or not self._worker.is_alive():
            return False
        if failed_scope is not None and not self._failed_scope_retry_available(user_id, failed_scope):
            # The caller is asking on behalf of a cached failure whose one retry
            # is already spent. Refusing here is what stops the loop: the worker
            # would otherwise pop the job, short-circuit, and still emit
            # projection_ready — which the client listens for and answers with
            # another /points, which asks again, at the worker's poll rate for
            # the life of the process.
            return False
        with self._projection_lock:
            if user_initiated:
                # A person asked for this, so give the scope a fresh budget:
                # without it /refresh answered `enqueued: true` while the worker
                # was guaranteed to short-circuit, and a user whose retry was
                # spent had no way back short of restarting the server. NOT for
                # a bare request — /points' staleness path passes none of this
                # and is driven by polling, so clearing there would hand a
                # poller the unlimited retries the budget exists to deny.
                self._failed_projection_scopes.pop(user_id, None)
            # Last writer wins: the scope reflects the requester's most recent
            # admin status, so a promotion/demotion between requests cannot
            # pin a stale scope.
            self._projection_requests[user_id] = all_images
        return True

    def start(self, invoker: "Invoker") -> None:
        self._invoker = invoker
        config = invoker.services.configuration
        if not config.image_index_enabled:
            return
        if self._worker is not None and self._worker.is_alive():
            # A second start would register duplicate image callbacks and race
            # a prior worker that outlived its stop() join.
            invoker.services.logger.warning("Image index service started twice; ignoring the second start")
            return

        # Projection state is per-run: a spent retry marker left over from the
        # previous run would deny a scope the retry this run's docstring
        # promises it, and a job requeued at the last shutdown would resurrect
        # here carrying an admin scope the user may no longer have.
        with self._projection_lock:
            self._failed_projection_scopes.clear()
            self._projection_requests.clear()

        device_override = config.image_index_device
        if device_override and not _is_cpu_override(device_override):
            # Ignored, as documented — but silently ignoring a near-miss like `CPU` leaves the
            # operator believing indexing runs on the CPU when it is following the model cache.
            invoker.services.logger.warning(
                f"image_index_device={device_override!r} is not understood and is being ignored; "
                "only 'cpu' selects CPU mode"
            )

        if self._encode_fn_override is not None:
            self._encode_fn = self._encode_fn_override
            self._model_id = self._model_id_override
        else:
            model_config = self._resolve_model_config(config.image_index_model)
            if model_config is None:
                invoker.services.logger.warning(self._model_not_installed_message(config.image_index_model))
                return
            self._model_config = model_config
            self._model_id = model_config.hash
            self._encode_fn = self._encode_with_model

        discarded = invoker.services.image_index_records.delete_embeddings_for_other_models(self._model_id)
        if discarded:
            invoker.services.logger.info(
                f"Discarded {discarded} image embeddings computed by a previously-configured model"
            )

        invoker.services.images.on_changed(self._on_image_changed)
        invoker.services.images.on_deleted(self._on_image_deleted)

        self._backfill_pending.set()
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="image_index_worker", daemon=True)
        self._worker.start()

    def stop(self, invoker: Optional["Invoker"] = None) -> None:
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=10)
            if self._worker.is_alive() and self._invoker is not None:
                self._invoker.services.logger.warning(
                    "Image index worker did not stop within 10s (likely mid-encode); abandoning daemon thread"
                )

    # --- Image service callbacks (caller's thread — enqueue and flag only, never I/O) ---

    def _on_image_changed(self, image_dto: ImageDTO) -> None:
        if image_dto.is_intermediate or image_dto.image_category != ImageCategory.GENERAL:
            # The image may have just LEFT eligibility (e.g. adopted as a
            # canvas asset). Drop any failure bookkeeping so a name no longer
            # counted in the totals cannot skew `failed` (and thus `pending`)
            # for the rest of the process. Flag only when something was
            # actually forgotten — this branch also fires for every
            # intermediate save during generation, which must stay silent.
            if image_dto.image_name in self._failed:
                self._failed.discard(image_dto.image_name)
                self._attempts.pop(image_dto.image_name, None)
                self._status_dirty.set()
            return
        with self._pending_lock:
            if image_dto.image_name in self._pending:
                return
            self._pending.add(image_dto.image_name)
        # Flagged before the enqueue so the worker reports the image as
        # pending before it starts (and certainly before it finishes) the
        # embed. The worker does the emit: callbacks stay free of DB reads
        # (which could block the generation thread behind the worker's own
        # long transactions), and single-threaded emission means events can
        # never be dispatched with out-of-order counts.
        self._status_dirty.set()
        self._queue.put(image_dto.image_name)

    def _on_image_deleted(self, image_name: str) -> None:
        # The DB row is removed by the images FK cascade; just forget local state.
        with self._pending_lock:
            self._pending.discard(image_name)
        self._failed.discard(image_name)
        self._attempts.pop(image_name, None)
        # A deletion gives the worker nothing to embed, so this flag is the
        # only signal clients get that the index shrank; the worker notices
        # within its poll interval. Bulk deletes coalesce into one emit.
        self._status_dirty.set()

    # --- Worker ---

    def _worker_loop(self) -> None:
        assert self._invoker is not None
        logger = self._invoker.services.logger
        while not self._stop_event.is_set():
            try:
                batch = self._next_batch()
                if self._status_dirty.is_set():
                    # The worker is the only emitter, so events are totally
                    # ordered by one thread and can never carry out-of-order
                    # counts. Checked after _next_batch so a just-enqueued
                    # image is reported pending before its embed runs.
                    self._status_dirty.clear()
                    try:
                        self._emit_status()
                    except Exception:
                        # For deletions this flag is the only signal clients
                        # get, so a failed emit must not consume it. Cleared
                        # before emitting (not after) so a set arriving
                        # mid-emit is never lost either.
                        self._status_dirty.set()
                        raise
                if batch is not None:
                    # Projections are serviced from INSIDE this wait (see
                    # _wait_for_idle_generation), not from a gate before it. A
                    # gate is only read once per pass, so a request arriving
                    # after the worker had already parked waited out the whole
                    # run — the exact "computing for the length of a generation"
                    # symptom it was added to fix.
                    self._wait_for_idle_generation()
                    if self._stop_event.is_set():
                        break
                    ok = self._process_batch(batch)
                    self._emit_status()
                    if ok:
                        continue
                    # Re-sweep so queue-sourced images that failed
                    # transiently are retried to _MAX_ATTEMPTS instead of
                    # lingering unembedded until the next restart.
                    self._backfill_pending.set()
                    # Back off after a failed batch so a systemic failure
                    # (broken model, OOM loop) cannot spin hot through the
                    # backlog re-fetching the same names. The delay grows
                    # with consecutive systemic failures: images are no
                    # longer retired by an outage, so without this an
                    # outage would retry the same batch at 1 Hz for as
                    # long as it lasts.
                    self._stop_event.wait(self._backoff_seconds())
                    if not self._systemic_failures:
                        # Per-image failures retire after _MAX_ATTEMPTS, so the
                        # backlog drains and quiescence arrives on its own.
                        continue
                    # A systemic failure charges no image, by design — which
                    # means this same batch reappears on every pass and
                    # quiescence never comes. Fall through to the projection
                    # work below rather than skipping it for the duration of
                    # the outage: a projection reads only already-stored
                    # embeddings, so it needs no encoder and is still correct
                    # while embedding is stalled. Without this the image map
                    # reports "computing" forever over a gallery that is in
                    # fact embedded.
                if self._pending_pokes and not self._backfill_pending.is_set():
                    # The index just went quiescent: poke each owner whose
                    # images were embedded since the last flush. Deferred to
                    # quiescence so a long backfill does not trigger a
                    # refetch (and possibly a projection recompute) per
                    # batch. Cleared only after all emits, so a raise here
                    # retries on the next pass (duplicate pokes are cheap).
                    for user_id in sorted(self._pending_pokes):
                        self._invoker.services.events.emit_image_index_updated(user_id=user_id)
                    self._pending_pokes.clear()
                    continue
                # No embedding work right now — projections run when the
                # index is quiescent so they cover the freshest state.
                if self._stop_event.is_set():
                    # Neither path that arrives here re-checks the flag: the
                    # quiescent fall-through never did, and the systemic-outage
                    # backoff wait returns instantly once stop is set. Starting
                    # a multi-minute fit now outlives stop()'s 10s join, so the
                    # abandoned thread writes the database and emits an event
                    # after the app believes the service is down.
                    break
                if self._vocab_build_requested.is_set():
                    # Cleared first: a request arriving mid-build wants the
                    # result of a build that is already covering it, and
                    # _build_vocab_embeddings is a no-op once one has landed.
                    self._vocab_build_requested.clear()
                    self._build_vocab_embeddings()
                    continue
                projection_job = self._next_projection_job()
                if projection_job is not None:
                    self._process_projection(*projection_job)
            except Exception:
                logger.exception("Unexpected error in the image index worker")
                # Whatever was in flight may have skipped its accounting, so
                # re-sweep for anything left unembedded and re-emit counts —
                # otherwise an image dropped here lingers with pending stuck
                # above zero until the next restart.
                self._backfill_pending.set()
                self._status_dirty.set()
                self._stop_event.wait(_POLL_SECONDS)

    def _next_batch(self) -> Optional[list[str]]:
        """Get the next batch of image names, preferring backfill work.

        Returns None when there is nothing to do right now.
        """
        assert self._invoker is not None
        batch_size = self._invoker.services.configuration.image_index_batch_size

        if self._backfill_pending.is_set():
            assert self._model_id is not None
            # Over-fetch by the failed count so permanently-failing images
            # cannot occlude the rest of the backlog.
            candidates = self._invoker.services.image_index_records.list_unembedded_image_names(
                self._model_id, limit=batch_size + len(self._failed)
            )
            batch = [name for name in candidates if name not in self._failed][:batch_size]
            if batch:
                return batch
            self._backfill_pending.clear()
            self._emit_status()

        try:
            first = self._queue.get(timeout=_POLL_SECONDS)
        except Empty:
            return None
        batch = [first]
        while len(batch) < batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break
        return batch

    def _process_batch(self, image_names: list[str]) -> bool:
        """Embed one batch. Returns False when anything in it failed."""
        assert self._invoker is not None
        assert self._encode_fn is not None
        assert self._model_id is not None
        logger = self._invoker.services.logger

        try:
            images: list[Image.Image] = []
            loaded_names: list[str] = []
            for name in image_names:
                try:
                    images.append(self._invoker.services.images.get_pil_image(name).convert("RGB"))
                    loaded_names.append(name)
                except Exception as e:
                    logger.warning(f"Image index: could not load '{name}' ({e})")
                    self._record_failure([name])

            if not loaded_names:
                return False

            try:
                embeddings = np.asarray(self._encode_fn(images), dtype=EMBEDDING_DTYPE)
            except Exception:
                logger.exception(f"Image index: failed to embed a batch of {len(loaded_names)} images")
                self._attribute_batch_failure("the encoder raised", loaded_names)
                return False

            if embeddings.ndim != 2 or embeddings.shape[0] != len(loaded_names):
                logger.error(f"Image index: encoder returned shape {embeddings.shape} for {len(loaded_names)} images")
                self._attribute_batch_failure("the encoder returned an unusable result", loaded_names)
                return False

            # L2-normalize so cosine similarity is a plain dot product downstream. The norm is
            # computed in float64: in float32 the sum of squares under/overflows well inside the
            # range of the vector itself (components uniformly below ~1e-22 give norm 0.0, above
            # ~1.8e19 give inf), which would misread a perfectly normalizable row as degenerate.
            norms = np.linalg.norm(embeddings.astype(np.float64), axis=1, keepdims=True)
            # A row whose norm is zero or non-finite cannot be normalized: it stays all-zero or
            # becomes non-finite, either of which the storage layer rejects because it yields NaN
            # in every similarity it takes part in. Drop those rows and fail only their own
            # names — the old `norms[norms == 0] = 1.0` handed an all-zero vector straight to
            # upsert_embedding, whose ValueError then failed the whole batch.
            degenerate = ~np.isfinite(norms).all(axis=1) | (norms[:, 0] == 0)
            if degenerate.any():
                dropped = [name for name, bad in zip(loaded_names, degenerate, strict=True) if bad]
                logger.warning(f"Image index: encoder returned unusable embeddings for {dropped}")
                self._record_failure(dropped)
                keep = ~degenerate
                # Reassigned before `stored` and the handler below, so dropped names are already
                # accounted for and cannot be failed a second time. Shrinking it also makes the
                # `len(loaded_names) == len(image_names)` return False, which re-arms the
                # backfill so these images are retried rather than stranded.
                loaded_names = [name for name, ok in zip(loaded_names, keep, strict=True) if ok]
                embeddings = embeddings[keep]
                norms = norms[keep]
            embeddings = (embeddings / norms).astype(np.float32)

            stored: list[str] = []
            try:
                for name, embedding in zip(loaded_names, embeddings, strict=True):
                    # Deliberately not caught per-image: any exception from here must reach the
                    # handler below, which fails every unstored name and returns False to re-arm
                    # the backfill. Swallowing one image's error here would leave it unembedded
                    # with no retry ever scheduled, wedging `pending` above zero.
                    self._invoker.services.image_index_records.upsert_embedding(name, self._model_id, embedding)
                    stored.append(name)
                    self._attempts.pop(name, None)
                    # An image that recovers (e.g. re-embedded after an
                    # update) must stop counting against the failed total.
                    self._failed.discard(name)
            except Exception:
                # A raise here (e.g. "database is locked") is a property of the database, not of
                # any image, so it is systemic: the unstored names stay pending and are retried
                # rather than being charged an attempt each.
                logger.exception(f"Image index: failed to store embeddings for a batch of {len(loaded_names)} images")
                self._record_systemic_failure(
                    "storing embeddings failed", [name for name in loaded_names if name not in stored]
                )
                return False
            finally:
                self._record_owner_pokes(stored)
                if stored:
                    # Anything stored proves the encoder and the database are both working, so
                    # the outage (if there was one) is over and the backoff resets. In the
                    # `finally` deliberately, i.e. reached even when the handler above has just
                    # counted a failure: a batch that stores ANY image is making progress, so
                    # the backlog drains and quiescence arrives on its own. Moving this to the
                    # success path instead left the counter with no reset while every batch
                    # partially failed, which escalated the wait to its 60s ceiling — a far
                    # worse outcome under mild write contention than the flat 1Hz retry it was
                    # meant to fix.
                    self._systemic_failures = 0
            return len(loaded_names) == len(image_names)
        finally:
            # Always release the batch from the dedup set — a name stuck in
            # _pending can never be re-enqueued by callbacks.
            self._forget_pending(image_names)

    def _attribute_batch_failure(self, reason: str, image_names: list[str]) -> None:
        """Charge a whole-batch embedding failure to the images or to the machinery."""
        if self._encoder_is_healthy():
            # The encoder works on a trivial image, so something about these images is what
            # broke it. Charge them, so a poisonous one is quarantined after _MAX_ATTEMPTS and
            # stops blocking the rest of the backlog.
            self._record_failure(image_names)
        else:
            self._record_systemic_failure(reason, image_names)

    def _encoder_is_healthy(self) -> bool:
        """Probe the encoder with a trivial image to tell a broken encoder from bad input.

        This is the discriminator the whole failure-attribution scheme rests on. A batch that
        fails to embed says nothing on its own about *why*: the model may be uninstalled, or one
        image in the batch may be corrupt. Repetition cannot separate the two — a real outage
        and a poisonous image both fail every time — so the encoder is asked directly.

        A healthy encoder means the batch failed because of its contents, and the images are
        charged an attempt so a bad one is eventually quarantined and the backfill can move past
        it. An unhealthy one means the machinery is down, and the images must not be charged.
        """
        assert self._encode_fn is not None
        try:
            probe = np.asarray(self._encode_fn([Image.new("RGB", (16, 16))]), dtype=EMBEDDING_DTYPE)
        except Exception:
            return False
        return probe.ndim == 2 and probe.shape[0] == 1 and probe.shape[1] > 0

    def _record_systemic_failure(self, reason: str, image_names: list[str]) -> None:
        """Record a failure of the machinery: back off, but charge no image for it.

        `_MAX_ATTEMPTS` exists so one bad file cannot spin the backfill forever. Charging an
        outage against that budget instead retires every image the outage touched, and because
        nothing clears `_failed` but a successful embed, restoring the model would not bring
        them back — only a restart would. The images stay pending here, so the sweep retries
        them once the machinery recovers, and `pending` keeps telling the truth meanwhile.
        """
        assert self._invoker is not None
        self._systemic_failures += 1
        self._invoker.services.logger.warning(
            f"Image index: {reason}; leaving {len(image_names)} image(s) pending for retry "
            f"(consecutive failures: {self._systemic_failures})"
        )

    def _backoff_seconds(self) -> float:
        if self._systemic_failures <= 0:
            return _POLL_SECONDS
        # Cap the exponent before the shift so a long outage cannot overflow it.
        return min(_POLL_SECONDS * (2 ** min(self._systemic_failures - 1, 16)), _MAX_BACKOFF_SECONDS)

    def _record_failure(self, image_names: list[str]) -> None:
        """Count a failure; move an image to the permanent-failure set only after repeated attempts."""
        for name in image_names:
            attempts = self._attempts.get(name, 0) + 1
            self._attempts[name] = attempts
            if attempts >= _MAX_ATTEMPTS:
                self._failed.add(name)

    def _forget_pending(self, image_names: list[str]) -> None:
        with self._pending_lock:
            self._pending.difference_update(image_names)

    # --- Projections ---

    def _failed_scope_retry_available(self, user_id: str, current_hash: str) -> bool:
        """Whether a cached projection that is empty over a non-empty gallery gets another fit.

        A fit that raised caches an EMPTY result stamped with the CURRENT scope
        hash. Both halves are deliberate — serving "empty" is honest, and the
        stamp stops a polling client from re-enqueueing a doomed recompute —
        but together with the hash short-circuit they make the failure
        terminal: /points answers `state: "empty", stale: false` (so the client
        stops asking) and every later job returns early (so the fit is never
        re-entered). The map stays blank until the accessible image set
        changes, and the row is in SQLite, so restarting does not clear it.

        Retrying a failed scope once splits the difference the original code
        could not: a transient failure — a MemoryError while a generation holds
        RAM, a JIT hiccup, the non-finite guard — heals on the next request,
        while a permanent one costs one wasted fit instead of one per poll.

        The bound is one retry per failed fit that reaches the cache, NOT one
        per (user, scope) for the life of the process: any later fit that
        produces points clears the marker, so a scope that fails, succeeds after
        a gallery change, and then fails again is retried again. That is the
        intent — the budget exists to stop a doomed fit from repeating, and a
        success in between is evidence the next failure is worth one more try.

        A pure check. The budget is spent in _process_projection, once the
        failed result is durably cached, so nothing that returns before the
        write — a "database is locked" on the embedding read, a shutdown, or a
        successful fit whose own write is lost — can burn a retry.
        """
        with self._projection_lock:
            return self._failed_projection_scopes.get(user_id) != current_hash

    def _next_projection_job(self) -> Optional[tuple[str, bool]]:
        with self._projection_lock:
            if not self._projection_requests:
                return None
            user_id = next(iter(self._projection_requests))
            return user_id, self._projection_requests.pop(user_id)

    def _process_projection(self, user_id: str, all_images: bool) -> None:
        assert self._invoker is not None
        assert self._model_id is not None
        logger = self._invoker.services.logger
        records = self._invoker.services.image_index_records

        scope_user = None if all_images else user_id
        try:
            names = records.list_accessible_embedded_images(scope_user, self._model_id)
            current_hash = scope_hash(self._model_id, names)
            cached = records.get_projection(user_id, self._model_id)
        except Exception:
            # The DB reads are as failure-prone as the fit ("database is
            # locked" is a designed-for condition here), and the job was
            # already popped from the dedup map. Re-queue it rather than let
            # the generic worker handler swallow it: /refresh has already
            # answered `enqueued: true`, and a client waiting on
            # image_map_projection_ready would otherwise wait forever.
            logger.exception(f"Image map: could not read the projection scope for user '{user_id}'; re-queueing")
            self._requeue_projection(user_id, all_images)
            return

        retrying_failed_scope = (
            cached is not None
            and cached.scope_hash == current_hash
            # "Failed" is *nothing servable*, not point_count == 0. /points
            # drops non-finite rows before it serves them, so a row with points
            # but no FINITE points is empty to every client while looking
            # populated here. Testing the count instead left that row failing
            # this check while /points kept asking on its behalf: the request
            # was granted, the branch below short-circuited without spending the
            # budget, the emit brought the client straight back, and the retry
            # that was supposed to break the cycle could never be spent. An
            # empty coords array reduces to False here too, so this subsumes the
            # count test rather than replacing it.
            and not np.isfinite(cached.coords).all(axis=1).any()
            and bool(names)
            and self._failed_scope_retry_available(user_id, current_hash)
        )
        if cached is not None and cached.scope_hash == current_hash and not retrying_failed_scope:
            # Nothing the projection depends on has changed, and the fit is
            # seeded, so recomputing would burn minutes of single-threaded CPU
            # to arrive at identical coordinates. Still emit, so a client that
            # asked for a refresh hears that its request was honoured.
            self._invoker.services.events.emit_image_map_projection_ready(
                user_id=user_id, point_count=cached.point_count
            )
            return

        try:
            # Read after the short-circuit, not before: this materializes every
            # accessible embedding (hundreds of MB on a large gallery), which is
            # pure waste on the no-op path that /points and /refresh drive.
            found_names, matrix = records.get_embeddings(names, self._model_id)
        except Exception:
            logger.exception(f"Image map: could not read embeddings for user '{user_id}'; re-queueing")
            self._requeue_projection(user_id, all_images)
            return

        if self._stop_event.is_set():
            # The fit is unbounded and stop() waits only 10s, so starting one
            # now means writing the database after the app has torn down.
            self._requeue_projection(user_id, all_images)
            return

        try:
            coords = compute_umap(matrix)
            if coords.shape[0] != len(found_names):
                raise RuntimeError(f"projection produced {coords.shape[0]} points for {len(found_names)} images")
            if not np.isfinite(coords).all():
                # A NaN reaching the cache makes /points raise inside sklearn
                # on every request, and the cache is only replaced when the
                # scope hash changes — so it would 500 that user indefinitely.
                # Treat it as a failed fit and take the empty-cache path.
                raise RuntimeError("projection produced non-finite coordinates")
        except Exception:
            logger.exception(f"Image map: UMAP projection failed for user '{user_id}' ({len(found_names)} points)")
            # Cache an empty projection under the CURRENT scope hash. Serving
            # "empty" is honest, and it stops the client from re-enqueueing a
            # doomed recompute on every poll. It does NOT make the failure
            # terminal: _failed_scope_retry_available grants this scope one more fit
            # per process, and the next gallery change flips the hash anyway.
            found_names = []
            coords = np.empty((0, 2), dtype=EMBEDDING_DTYPE)

        # Hash the scope the projection was computed against (on failure the
        # empty cache still claims the current scope, so staleness detection —
        # not client polling — decides when to retry).
        try:
            records.set_projection(
                user_id,
                self._model_id,
                current_hash,
                projection_params(n_points=len(found_names)),
                found_names,
                coords,
            )
        except Exception:
            # Same reasoning as the read above: the fit is done but unsaved, so
            # re-queue rather than drop it and leave /refresh's promise unmet.
            logger.exception(f"Image map: could not cache the projection for user '{user_id}'; re-queueing")
            self._requeue_projection(user_id, all_images)
            return

        # The budget moves only once the outcome is durable. Spending it before
        # the fit looked safer — nothing between the spend and compute_umap can
        # return — but it ignored the write: a fit that SUCCEEDS and then loses
        # its `set_projection` to a locked database returns above this line, and
        # the requeued job finds the old empty row with the budget already gone.
        # That threw away minutes of correct work and left the map permanently
        # blank without a single failed fit anywhere in the sequence. Spending
        # here means the budget is charged for exactly what it is meant to
        # bound: a failed fit that reached the cache.
        with self._projection_lock:
            if found_names:
                # A fit that produced points clears the marker, so a LATER
                # failure over some future scope gets its own retry rather than
                # inheriting this one's spent budget.
                self._failed_projection_scopes.pop(user_id, None)
            elif retrying_failed_scope:
                self._failed_projection_scopes[user_id] = current_hash
        try:
            self._invoker.services.events.emit_image_map_projection_ready(user_id=user_id, point_count=len(found_names))
        except Exception:
            # The row is cached, so a polling client recovers on its own; only
            # an event-driven client would hang. Log rather than unwind into
            # the generic worker handler, which would re-sweep the backfill for
            # what is purely a delivery failure.
            logger.exception(f"Image map: could not emit projection_ready for user '{user_id}'")

    def _requeue_projection(self, user_id: str, all_images: bool) -> None:
        """Put a popped job back, without clobbering a newer request for that user."""
        with self._projection_lock:
            self._projection_requests.setdefault(user_id, all_images)

    def _wait_for_idle_generation(self) -> None:
        """Block until no generation is in progress, unless embedding on CPU.

        The embedding model shares the model cache (and therefore VRAM) with
        generation; embedding on the GPU mid-generation would thrash both.

        Projection jobs are serviced while parked here. A projection reads
        stored embeddings only — no encoder, no GPU — so it is exactly the work
        that can proceed while embedding cannot, and this is the only place it
        runs alongside pending embedding work: everywhere else projections are
        deferred to quiescence, because a fit between backfill batches is
        obsolete as soon as the next batch stores. Nothing new can embed while
        we are parked, so the accessible set holds still and a repeat request
        short-circuits on its scope hash rather than refitting.
        """
        assert self._invoker is not None
        if self._cpu_mode():
            return
        session_queue = self._invoker.services.session_queue
        if session_queue is None:
            return
        while not self._stop_event.is_set():
            try:
                status = session_queue.get_queue_status(DEFAULT_QUEUE_ID)
            except Exception:
                return
            if status.in_progress == 0:
                return
            projection_job = self._next_projection_job()
            if projection_job is not None:
                # No wait before looping: a request that arrived during the fit
                # is served immediately, and the generation check above is cheap.
                self._process_projection(*projection_job)
                continue
            self._stop_event.wait(_POLL_SECONDS)

    def _emit_status(self) -> None:
        assert self._invoker is not None
        status = self.get_status()
        if status is None:
            return
        # `failed` travels with the counts because it is the only thing that explains a
        # settled index that is not complete: `pending` excludes failures, so without it a
        # consumer sees pending == 0 with embedded < total and cannot tell "done" from
        # "gave up on some".
        self._invoker.services.events.emit_image_index_status(
            total=status.total, embedded=status.embedded, pending=status.pending, failed=status.failed
        )

    def _record_owner_pokes(self, image_names: list[str]) -> None:
        """Queue a per-user poke for each owner of freshly stored embeddings.

        The status event is admin-only (its counts aggregate every user's
        images), so without this, non-admin clients would never hear that
        their own generations reached the index. Owner lookups are DB reads,
        which is fine here: this runs on the worker thread.
        """
        assert self._invoker is not None
        for name in image_names:
            try:
                user_id = self._invoker.services.image_records.get_user_id(name)
            except Exception:
                continue
            if user_id:
                self._pending_pokes.add(user_id)

    # --- Model handling ---

    def _model_not_installed_message(self, model_name: str) -> str:
        assert self._invoker is not None
        # The starter catalog ships same-named models of different types (the
        # 'clip-vit-large-patch14' CLIP-L text encoder vs. the CLIP Vision
        # image encoder), so name the type mismatch when that is the problem.
        # search_by_attr drops the name filter for a falsy name and would
        # return every installed model.
        same_name = (
            self._invoker.services.model_manager.store.search_by_attr(model_name=model_name) if model_name else []
        )
        if same_name:
            types = ", ".join(sorted({config.type.value for config in same_name}))
            return (
                f"Image indexing is enabled, but the installed model named '{model_name}' is of type "
                f"'{types}', not a CLIP Vision or SigLIP image encoder. Install the image-encoder model of "
                "the same name (for the default, the 'CLIP ViT-L Image Encoder' starter model from source "
                "'InvokeAI/clip-vit-large-patch14'). The image index will not be updated."
            )
        return (
            f"Image indexing is enabled but the embedding model '{model_name}' is not installed "
            "(expected a CLIP Vision or SigLIP model). The image index will not be updated."
        )

    def _resolve_model_config(self, model_name: str) -> Optional["AnyModelConfig"]:
        assert self._invoker is not None
        if not model_name:
            # `search_by_attr` drops the name predicate for a falsy name and would return every
            # installed model of the type, so an empty config value would silently adopt an
            # arbitrary one — and then `start()` discards every embedding computed by the model
            # the user actually meant. `_model_not_installed_message` guards this too.
            return None
        store = self._invoker.services.model_manager.store
        for model_type in (ModelType.CLIPVision, ModelType.SigLIP):
            configs = store.search_by_attr(model_name=model_name, model_type=model_type)
            if not configs:
                continue
            # Names are not unique (the UNIQUE(name, base, type) constraint is gone), and
            # `search_by_attr` orders by type/base/name/format — so for two otherwise identical
            # rows the tiebreak is insertion order, and reinstalling one moves the winner. That
            # changes the model hash, and `start()` then discards every embedding computed under
            # the old one. Order by key instead: stable across reinstalls of the same set.
            chosen = min(configs, key=lambda config: config.key)
            if len(configs) > 1:
                self._invoker.services.logger.warning(
                    f"Multiple {model_type.value} models named '{model_name}' are installed; "
                    f"using '{chosen.key}' for the image index"
                )
            return chosen
        return None

    def _cpu_mode(self) -> bool:
        """True when embeddings should be computed on a service-local CPU copy of the model.

        Both callers only ever ask whether this is CPU mode — the GPU path follows whatever
        device the model cache picked. So the override is never turned into a `torch.device`:
        `image_index_device` is a free-form config string with no validator, and
        `torch.device("CPU")` raises. Building one here put the worker in a permanent
        exception loop on a typo, bypassing the `_MAX_ATTEMPTS` bound because it raised before
        any batch was attempted.
        """
        override = self._invoker.services.configuration.image_index_device if self._invoker else None
        if _is_cpu_override(override):
            return True
        # Any other value is ignored, as the config documents. Fall back to the auto-chosen
        # device so a CPU-only host still gets CPU mode.
        return TorchDevice.choose_torch_device().type == "cpu"

    def _model_abs_path(self) -> Path:
        assert self._invoker is not None
        assert self._model_config is not None
        model_path = Path(self._model_config.path)
        if model_path.is_absolute():
            return model_path
        return self._invoker.services.configuration.models_path / model_path

    def _get_processor(self) -> Any:
        # embed_image runs on request threads concurrently with the worker,
        # so the lazy init must be guarded (like _get_text_encoder).
        with self._vision_init_lock:
            if self._processor is None:
                from transformers import CLIPImageProcessor, SiglipImageProcessor

                assert self._model_config is not None
                processor_cls = (
                    SiglipImageProcessor if self._model_config.type is ModelType.SigLIP else CLIPImageProcessor
                )
                try:
                    self._processor = processor_cls.from_pretrained(str(self._model_abs_path()), local_files_only=True)
                except OSError:
                    # InvokeAI-published CLIP Vision model dirs ship no
                    # preprocessor_config.json; IP-Adapter and FLUX construct the
                    # processor with defaults for the same reason.
                    self._processor = processor_cls()
            return self._processor

    def _embed(self, model: Any, images: list[Image.Image], device: torch.device) -> np.ndarray:
        from transformers import CLIPVisionModelWithProjection, SiglipVisionModel

        with torch.no_grad():
            inputs = self._get_processor()(images=images, return_tensors="pt")
            # The cached model may be fp16 on GPU; inputs must match its dtype.
            pixel_values = inputs["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype)
            if isinstance(model, SiglipVisionModel):
                embeddings = model(pixel_values=pixel_values).pooler_output
            elif isinstance(model, CLIPVisionModelWithProjection):
                embeddings = model(pixel_values=pixel_values).image_embeds
            else:
                raise TypeError(f"Unsupported embedding model class: {type(model).__name__}")
            return embeddings.float().cpu().numpy()

    def _encode_with_model(self, images: list[Image.Image]) -> np.ndarray:
        assert self._invoker is not None
        assert self._model_config is not None

        if self._cpu_mode():
            # CPU mode exists to avoid touching the model cache (and VRAM) at
            # all, so it keeps its own RAM-resident copy of the model. The
            # lazy init is guarded: embed_image calls in from request threads.
            with self._vision_init_lock:
                if self._cpu_model is None:
                    from transformers import CLIPVisionModelWithProjection, SiglipVisionModel

                    model_path = str(self._model_abs_path())
                    model_cls = (
                        SiglipVisionModel
                        if self._model_config.type is ModelType.SigLIP
                        else CLIPVisionModelWithProjection
                    )
                    # Process-global patch, so it needs the process-global lock —
                    # see _get_text_encoder for what goes wrong without it.
                    with MODEL_LOAD_LOCK.write_lock(), skip_torch_weight_init():
                        model = model_cls.from_pretrained(model_path, local_files_only=True)
                    model.eval()
                    self._cpu_model = model
            return self._embed(self._cpu_model, images, torch.device("cpu"))

        loaded = self._invoker.services.model_manager.load.load_model(self._model_config)
        with loaded.model_on_device() as (_, model):
            # The cache decides where the model actually lives; follow it.
            device = next(model.parameters()).device
            return self._embed(model, images, device)
