from __future__ import annotations

import json
import os
import secrets
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image


class RemoteInvokeError(RuntimeError):
    """Raised when communication with the remote InvokeAI worker fails."""


@dataclass(frozen=True)
class RemoteCredentials:
    email: str
    password: str
    remember_me: bool = True


@dataclass(frozen=True)
class RemoteConfig:
    base_url: str
    api_key: str
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    verify_ssl: bool = True
    credentials_file: str = ""
    user_id: str = ""

    @classmethod
    def from_environment(cls, base_url: str = "", verify_ssl: bool = True, user_id: str = "") -> "RemoteConfig":
        url = (base_url or os.getenv("INVOKE_REMOTE_URL", "")).strip().rstrip("/")
        if not url:
            raise RemoteInvokeError(
                "Remote InvokeAI URL is not configured. Set INVOKE_REMOTE_URL or fill in Remote URL on the node."
            )

        default_credentials_file = Path(__file__).with_name("remote_auth.json")
        credentials_file = os.getenv("INVOKE_REMOTE_AUTH_FILE", str(default_credentials_file)).strip()

        return cls(
            base_url=url,
            api_key=os.getenv("INVOKE_REMOTE_API_KEY", "").strip(),
            auth_header=os.getenv("INVOKE_REMOTE_AUTH_HEADER", "Authorization").strip() or "Authorization",
            auth_prefix=os.getenv("INVOKE_REMOTE_AUTH_PREFIX", "Bearer "),
            verify_ssl=verify_ssl,
            credentials_file=credentials_file,
            user_id=user_id,
        )


class RemoteInvokeClient:
    def __init__(self, config: RemoteConfig, request_timeout_seconds: float = 30.0):
        self.config = config
        self.request_timeout_seconds = request_timeout_seconds
        self._auth_checked = False
        self._multiuser = False
        self._token = config.api_key or ""
        self._credentials: RemoteCredentials | None = None

    def _ssl_context(self):
        if self.config.verify_ssl:
            return None
        return ssl._create_unverified_context()  # noqa: SLF001 - explicitly user-controlled for LAN/self-signed use

    @staticmethod
    def _normalise_url(url: str) -> str:
        return str(url).strip().rstrip("/")

    def _load_credentials(self) -> RemoteCredentials:
        if self._credentials is not None:
            return self._credentials

        if self.config.user_id:
            # Per-local-user saved credentials; import lazily during graph bootstrap.
            from invokeai.app.invocations.remote_worker.credential_vault import get_saved_credentials

            entry = get_saved_credentials(self.config.user_id, self.config.base_url)
            if entry is None:
                raise RemoteInvokeError(
                    f"Remote InvokeAI at {self.config.base_url} requires login. "
                    "Open Remote Workers, enter this worker's email and password, and save them."
                )
            self._credentials = RemoteCredentials(
                email=str(entry["email"]),
                password=str(entry["password"]),
                remember_me=bool(entry.get("remember_me", True)),
            )
            return self._credentials

        # Legacy file path for non-panel callers without a local user identity.
        path = Path(self.config.credentials_file)
        if not path.is_file():
            raise RemoteInvokeError(
                "Remote InvokeAI has multi-user mode enabled, but no credentials file was found. "
                f"Create '{path}' from remote_auth.example.json and add credentials for {self.config.base_url}, "
                "or set INVOKE_REMOTE_AUTH_FILE to another JSON file."
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RemoteInvokeError(f"Could not read remote auth file '{path}': {exc}") from exc

        if not isinstance(data, dict):
            raise RemoteInvokeError(f"Remote auth file '{path}' must contain a JSON object")

        servers = data.get("servers", data)
        if not isinstance(servers, dict):
            raise RemoteInvokeError(f"Remote auth file '{path}' must contain a 'servers' object")

        wanted = self._normalise_url(self.config.base_url)
        entry = None
        for key, value in servers.items():
            if self._normalise_url(str(key)) == wanted:
                entry = value
                break

        if not isinstance(entry, dict):
            raise RemoteInvokeError(
                f"Remote InvokeAI has multi-user mode enabled, but '{path}' has no credentials for {wanted}"
            )

        email = str(entry.get("email", "")).strip()
        password = str(entry.get("password", ""))
        remember_me = bool(entry.get("remember_me", True))
        if not email or not password:
            raise RemoteInvokeError(
                f"Credentials for {wanted} in '{path}' must include non-empty 'email' and 'password' values"
            )

        self._credentials = RemoteCredentials(email=email, password=password, remember_me=remember_me)
        return self._credentials

    def _headers(self, json_body: bool = False, include_auth: bool = True) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if include_auth and self._token:
            headers[self.config.auth_header] = f"{self.config.auth_prefix}{self._token}"
        return headers

    def _request_raw(
        self,
        method: str,
        path: str,
        body: Any = None,
        json_body: bool = False,
        include_auth: bool = True,
        content_type: str | None = None,
        content_length: int | None = None,
    ) -> bytes:
        url = f"{self.config.base_url}{path}"
        headers = self._headers(json_body=json_body, include_auth=include_auth)
        if content_type is not None:
            headers["Content-Type"] = content_type
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
                context=self._ssl_context(),
            ) as response:
                return response.read()
        except urllib.error.URLError as exc:
            if isinstance(exc, urllib.error.HTTPError):
                raise
            raise RemoteInvokeError(f"Could not reach remote InvokeAI at {url}: {exc}") from exc

    def _login(self) -> None:
        credentials = self._load_credentials()
        payload = json.dumps(
            {
                "email": credentials.email,
                "password": credentials.password,
                "remember_me": credentials.remember_me,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            raw = self._request_raw(
                "POST",
                "/api/v1/auth/login",
                body=payload,
                json_body=True,
                include_auth=False,
            )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise RemoteInvokeError(
                f"Remote InvokeAI login failed with HTTP {exc.code} for {self.config.base_url}: {detail[:2000]}"
            ) from exc

        try:
            data = json.loads(raw.decode("utf-8"))
            token = str(data.get("token", "")).strip() if isinstance(data, dict) else ""
        except Exception as exc:
            raise RemoteInvokeError("Remote InvokeAI login returned invalid JSON") from exc
        if not token:
            raise RemoteInvokeError("Remote InvokeAI login succeeded but did not return a token")
        self._token = token

    def _ensure_auth_mode(self) -> None:
        if self._auth_checked:
            if self._multiuser and not self._token:
                self._login()
            return

        # InvokeAI v7 always exposes /api/v1/auth/status. The response explicitly
        # tells us whether multi-user mode is enabled; endpoint existence alone is
        # not a valid signal.
        try:
            raw = self._request_raw("GET", "/api/v1/auth/status", include_auth=False)
        except urllib.error.HTTPError as exc:
            # Keep a compatibility fallback for unusual v7 builds/proxies that do not
            # expose the status endpoint: treat a missing endpoint as single-user.
            if exc.code in (404, 405):
                self._multiuser = False
                self._auth_checked = True
                self._token = self.config.api_key or ""
                return
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise RemoteInvokeError(
                f"Could not determine remote InvokeAI authentication mode: HTTP {exc.code} from "
                f"/api/v1/auth/status: {detail[:2000]}"
            ) from exc

        try:
            status = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RemoteInvokeError("Remote InvokeAI auth status returned invalid JSON") from exc
        if not isinstance(status, dict) or "multiuser_enabled" not in status:
            raise RemoteInvokeError("Remote InvokeAI auth status did not include the v7 'multiuser_enabled' field")

        self._multiuser = bool(status.get("multiuser_enabled"))
        self._auth_checked = True
        if self._multiuser and bool(status.get("setup_required")):
            raise RemoteInvokeError(
                f"Remote InvokeAI at {self.config.base_url} has multi-user mode enabled but initial admin setup is still required"
            )
        if self._multiuser and not self._token:
            self._login()

    def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        json_body: bool = False,
        content_type: str | None = None,
        content_length: int | None = None,
    ) -> bytes:
        self._ensure_auth_mode()
        try:
            return self._request_raw(
                method,
                path,
                body=body,
                json_body=json_body,
                include_auth=True,
                content_type=content_type,
                content_length=content_length,
            )
        except urllib.error.HTTPError as exc:
            # In multi-user mode, a 401 usually means the cached JWT expired. Re-login
            # once and replay the exact request. Do not loop indefinitely on bad credentials.
            if exc.code == 401 and self._multiuser:
                self._token = ""
                self._login()
                try:
                    return self._request_raw(
                        method,
                        path,
                        body=body,
                        json_body=json_body,
                        include_auth=True,
                        content_type=content_type,
                        content_length=content_length,
                    )
                except urllib.error.HTTPError as retry_exc:
                    exc = retry_exc
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise RemoteInvokeError(f"Remote InvokeAI returned HTTP {exc.code} for {path}: {detail[:2000]}") from exc

    def _request_json_value(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        json_body = payload is not None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        raw = self._request(method, path, body=body, json_body=json_body)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            preview = raw[:500].decode("utf-8", errors="replace")
            raise RemoteInvokeError(f"Expected JSON from remote InvokeAI for {path}, got: {preview}") from exc

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._request_json_value(method, path, payload)
        if not isinstance(data, dict):
            raise RemoteInvokeError(f"Expected a JSON object from remote InvokeAI for {path}")
        return data

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request_json_value("GET", "/api/v2/models/")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("models", "items", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        raise RemoteInvokeError("Remote /api/v2/models/ response did not contain a model list")

    def get_model(self, key: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(str(key), safe="")
        return self._request_json("GET", f"/api/v2/models/i/{encoded}")

    def get_model_layout(self, key: str) -> dict[str, Any]:
        """Return the Remote Workers layout signature for one registered model."""
        encoded = urllib.parse.quote(str(key), safe="")
        return self._request_json("GET", f"/api/v1/remote_workers/models/{encoded}/layout")

    def get_model_by_hash(self, model_hash: str) -> dict[str, Any] | None:
        """Return the remote model record with this content hash, or None when it is absent."""
        encoded = urllib.parse.quote(str(model_hash), safe="")
        try:
            return self._request_json("GET", f"/api/v2/models/get_by_hash?hash={encoded}")
        except RemoteInvokeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def install_model_from_url(self, source_url: str, *, name: str = "") -> dict[str, Any]:
        """Ask the remote InvokeAI model manager to download/probe/register one model URL."""
        encoded = urllib.parse.quote(str(source_url), safe="")
        config: dict[str, Any] = {}
        if name.strip():
            config["name"] = name.strip()
        return self._request_json(
            "POST",
            f"/api/v2/models/install?source={encoded}&inplace=false",
            payload=config,
        )

    def install_directory_from_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Wait briefly if the previous cancelled transfer is still cleaning up."""
        deadline = time.monotonic() + 20
        while True:
            try:
                return self._request_json("POST", "/api/v1/remote_workers/diffusers/install", payload=manifest)
            except RemoteInvokeError as exc:
                if (
                    "HTTP 422" not in str(exc)
                    or "Previous transfer cancellation is still being cleaned up" not in str(exc)
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(0.25)

    def get_directory_install_job(self, job_id: int) -> dict[str, Any]:
        return self._request_json("GET", f"/api/v1/remote_workers/diffusers/install/{int(job_id)}")

    def cancel_model_install(self, job_id: int, *, directory: bool = False) -> None:
        """Stop one install job; never delete a completed installed model."""
        path = (
            f"/api/v1/remote_workers/diffusers/install/{int(job_id)}"
            if directory
            else f"/api/v2/models/install/{int(job_id)}"
        )
        self._request("DELETE", path)

    def get_model_install_job(self, job_id: int) -> dict[str, Any]:
        return self._request_json("GET", f"/api/v2/models/install/{int(job_id)}")

    @staticmethod
    def _model_payload(detail: dict[str, Any]) -> dict[str, Any]:
        nested = detail.get("model")
        return nested if isinstance(nested, dict) else detail

    def remap_model_identifiers(
        self,
        graph: dict[str, Any],
        missing_model_handler: Callable[[dict[str, Any]], None] | None = None,
        model_match_validator: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    ) -> list[str]:
        """Replace local model keys with the matching remote installation keys.

        Resolve by model hash first. The hash is portable across InvokeAI installs,
        while the database key/UUID is installation-local. Name+base+type remains a
        compatibility fallback for model identifiers without a usable hash.

        If ``missing_model_handler`` is supplied, it is called once for a required model
        that cannot be found remotely. The handler may transfer/install the model; resolution
        is then retried before the graph is rejected.

        ``model_match_validator`` can reject a same-hash candidate when installation-local
        details (such as directory layout) are incompatible with the primary model.
        """
        remote_models = self.list_models()
        details_cache: dict[str, dict[str, Any]] = {}
        messages: list[str] = []
        handled_missing: set[str] = set()

        def detail_for(model: dict[str, Any]) -> dict[str, Any]:
            key = str(model.get("key", ""))
            if not key:
                return model
            if key not in details_cache:
                try:
                    details_cache[key] = self.get_model(key)
                except RemoteInvokeError:
                    details_cache[key] = model
            return details_cache[key]

        def refresh_models() -> None:
            nonlocal remote_models
            remote_models = self.list_models()
            details_cache.clear()

        def identifier_from(detail: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
            detail = self._model_payload(detail)
            result = deepcopy(fallback)
            for k in ("key", "hash", "name", "base", "type", "submodel_type"):
                if k in detail and detail[k] is not None:
                    result[k] = detail[k]
            return result

        def candidate_is_compatible(value: dict[str, Any], detail: dict[str, Any]) -> bool:
            local_hash = str(value.get("hash") or "").strip()
            if model_match_validator is None or not local_hash:
                return True
            return bool(model_match_validator(deepcopy(value), self._model_payload(detail)))

        def resolve(value: dict[str, Any]) -> dict[str, Any] | None:
            local_hash = str(value.get("hash") or "").strip()
            if local_hash:
                # Do not use /get_by_hash here: that endpoint returns only the first
                # record, but multiple installs can legitimately share a weight hash
                # while having different directory layouts.
                for model in remote_models:
                    payload = self._model_payload(model)
                    if str(payload.get("hash") or "").strip() != local_hash:
                        continue
                    detail = detail_for(model)
                    if candidate_is_compatible(value, detail):
                        return detail

            local_name = value.get("name")
            local_base = value.get("base")
            local_type = value.get("type")
            candidates = [
                model
                for model in remote_models
                if self._model_payload(model).get("name") == local_name
                and self._model_payload(model).get("base") == local_base
                and self._model_payload(model).get("type") == local_type
            ]
            verified: list[dict[str, Any]] = []
            for candidate in candidates:
                detail = detail_for(candidate)
                payload = self._model_payload(detail)
                remote_hash = str(payload.get("hash") or "").strip()
                if local_hash and remote_hash and local_hash != remote_hash:
                    continue
                if not candidate_is_compatible(value, detail):
                    continue
                verified.append(detail)

            if len(verified) == 1:
                return verified[0]
            if len(verified) > 1:
                raise RemoteInvokeError(
                    f"Remote worker has multiple matching models named '{local_name}'. "
                    "Hash-based resolution could not disambiguate them."
                )
            return None

        def remap(value: Any) -> Any:
            if isinstance(value, list):
                return [remap(x) for x in value]
            if not isinstance(value, dict):
                return value

            if all(k in value for k in ("key", "name", "base", "type")):
                local_name = str(value.get("name") or value.get("key") or "model")
                local_hash = str(value.get("hash") or "").strip()
                resolved = resolve(value)

                missing_identity = local_hash or str(value.get("key") or local_name)
                if resolved is None and missing_model_handler is not None and missing_identity not in handled_missing:
                    handled_missing.add(missing_identity)
                    missing_model_handler(deepcopy(value))
                    refresh_models()
                    resolved = resolve(value)

                if resolved is None:
                    if local_hash:
                        raise RemoteInvokeError(
                            f"Remote worker does not have required model '{local_name}' with hash {local_hash}"
                        )
                    raise RemoteInvokeError(
                        f"Remote worker does not have required model '{local_name}' "
                        f"(base={value.get('base')}, type={value.get('type')})"
                    )

                old_key = str(value.get("key"))
                mapped = identifier_from(resolved, value)
                new_key = str(mapped.get("key"))
                remote_hash = str(mapped.get("hash") or "").strip()
                if local_hash and remote_hash and local_hash != remote_hash:
                    raise RemoteInvokeError(
                        f"Remote model '{local_name}' resolved to hash {remote_hash}, expected {local_hash}"
                    )
                if new_key != old_key:
                    method = "hash" if local_hash else "name/base/type"
                    messages.append(f"{local_name}: {old_key} -> {new_key} ({method})")
                return mapped

            return {k: remap(v) for k, v in value.items()}

        nodes = graph.get("nodes")
        if not isinstance(nodes, dict):
            raise RemoteInvokeError("Executable graph has no nodes object")
        for node_id, node in list(nodes.items()):
            nodes[node_id] = remap(node)
        return messages

    def cancel_queue_item(self, item_id: int | str, queue_id: str = "default") -> dict[str, Any]:
        """Cancel only this remote queue item, with this client's saved user credentials."""
        queue_path = urllib.parse.quote(queue_id, safe="")
        return self._request_json("PUT", f"/api/v1/queue/{queue_path}/i/{int(item_id)}/cancel")

    def delete_queue_item(self, item_id: int | str, queue_id: str = "default") -> None:
        """Remove one finished worker queue record; do not prune anyone else's history."""
        queue_path = urllib.parse.quote(queue_id, safe="")
        self._request("DELETE", f"/api/v1/queue/{queue_path}/i/{int(item_id)}")

    def get_item(self, item_id: int | str, queue_id: str = "default") -> dict[str, Any]:
        return self._request_json("GET", f"/api/v1/queue/{urllib.parse.quote(queue_id)}/i/{int(item_id)}")

    def get_current_item(self, queue_id: str = "default") -> dict[str, Any] | None:
        data = self._request_json_value("GET", f"/api/v1/queue/{urllib.parse.quote(queue_id)}/current")
        if data is None:
            return None
        if not isinstance(data, dict):
            raise RemoteInvokeError("Expected current queue item to be a JSON object or null")
        return data

    def list_progress_previews(self, queue_id: str = "default") -> list[dict[str, Any]]:
        """Return InvokeAI v7's retained live progress previews for this authenticated user."""
        data = self._request_json_value("GET", f"/api/v1/queue/{urllib.parse.quote(queue_id)}/previews")
        if not isinstance(data, list):
            raise RemoteInvokeError("Remote progress previews endpoint did not return a JSON list")
        return [entry for entry in data if isinstance(entry, dict)]

    def get_progress_preview(self, item_id: int, queue_id: str = "default") -> dict[str, Any] | None:
        """Return the latest retained progress preview for one remote queue item, if available."""
        wanted = int(item_id)
        for entry in self.list_progress_previews(queue_id=queue_id):
            try:
                if int(entry.get("item_id")) == wanted:
                    return entry
            except (TypeError, ValueError):
                continue
        return None

    def enqueue_graph(
        self,
        graph: dict[str, Any],
        queue_id: str = "default",
        origin: str = "invokeai-remote-worker-node",
        destination: str | None = None,
        workflow: dict[str, Any] | None = None,
    ) -> int:
        batch: dict[str, Any] = {
            "graph": graph,
            "runs": 1,
            "origin": origin,
        }
        if destination is not None:
            batch["destination"] = destination
        if workflow is not None:
            batch["workflow"] = workflow
        payload = {"batch": batch}
        result = self._request_json(
            "POST",
            f"/api/v1/queue/{urllib.parse.quote(queue_id)}/enqueue_batch",
            payload=payload,
        )
        item_ids = result.get("item_ids")
        if not isinstance(item_ids, list) or not item_ids:
            raise RemoteInvokeError(f"Remote enqueue succeeded but no item_ids were returned: {result}")
        return int(item_ids[0])

    def wait_for_item(
        self,
        item_id: int,
        queue_id: str = "default",
        poll_interval_seconds: float = 0.75,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            item = self.get_item(item_id=item_id, queue_id=queue_id)
            status = str(item.get("status", "")).lower()
            if status == "completed":
                return item
            if status in {"failed", "canceled", "cancelled"}:
                errors = item.get("session", {}).get("errors", {})
                raise RemoteInvokeError(
                    f"Remote queue item {item_id} ended with status '{status}'. Errors: {json.dumps(errors)[:3000]}"
                )
            if time.monotonic() - started > timeout_seconds:
                raise RemoteInvokeError(f"Remote queue item {item_id} timed out after {timeout_seconds:g} seconds")
            time.sleep(poll_interval_seconds)

    @staticmethod
    def extract_image_names(
        item: dict[str, Any],
        non_intermediate_only: bool = True,
        allowed_result_node_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        allow_empty: bool = False,
    ) -> list[str]:
        """Return unique image names from completed session results.

        If ``allowed_result_node_ids`` is provided, only results from those exact source
        graph nodes are considered. Remote result filtering uses this to respect InvokeAI's
        Save to Gallery setting (``is_intermediate == False``) instead of importing every
        temporary image produced by a multi-stage workflow.

        Without an explicit allow-list, intermediate image-producing nodes are skipped by
        default. The legacy fallback to all image outputs is retained only in that mode.
        """
        session = item.get("session", {})
        results = session.get("results", {}) if isinstance(session, dict) else {}
        if not isinstance(results, dict):
            raise RemoteInvokeError("Remote queue item has no session.results object")

        graph_nodes = {}
        graph = session.get("graph") if isinstance(session, dict) else None
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), dict):
            graph_nodes = graph["nodes"]

        allowed_ids = None if allowed_result_node_ids is None else {str(x) for x in allowed_result_node_ids}

        def collect(filter_intermediate: bool) -> list[str]:
            image_names: list[str] = []
            seen: set[str] = set()
            for result_id, result in results.items():
                if not isinstance(result, dict):
                    continue
                if allowed_ids is not None and str(result_id) not in allowed_ids:
                    continue
                if filter_intermediate and graph_nodes:
                    graph_node = graph_nodes.get(str(result_id))
                    if isinstance(graph_node, dict) and bool(graph_node.get("is_intermediate", False)):
                        continue

                candidates: list[str] = []
                image = result.get("image")
                if isinstance(image, dict) and image.get("image_name"):
                    candidates.append(str(image["image_name"]))
                images = result.get("images")
                if isinstance(images, list):
                    for entry in images:
                        if isinstance(entry, dict) and entry.get("image_name"):
                            candidates.append(str(entry["image_name"]))

                for name in candidates:
                    if name not in seen:
                        seen.add(name)
                        image_names.append(name)
            return image_names

        image_names = collect(non_intermediate_only)
        # Do not fall back to temporary images when the caller supplied an explicit
        # Save-to-Gallery allow-list. Importing extra intermediate images would violate
        # the workflow's gallery settings.
        if not image_names and non_intermediate_only and allowed_ids is None:
            image_names = collect(False)
        if not image_names and not allow_empty:
            if allowed_ids is not None:
                raise RemoteInvokeError(
                    "Remote render completed but none of the Save-to-Gallery nodes produced an image output"
                )
            raise RemoteInvokeError("Remote render completed but no image output image_name was found")
        return image_names

    @staticmethod
    def extract_video_names(item: dict[str, Any]) -> list[str]:
        """Unique video outputs from completed InvokeAI v7 sessions (including video lists)."""
        session = item.get("session", {})
        results = session.get("results", {}) if isinstance(session, dict) else {}
        if not isinstance(results, dict):
            raise RemoteInvokeError("Remote queue item has no session.results object")
        names: list[str] = []
        seen: set[str] = set()
        for result in results.values():
            if not isinstance(result, dict):
                continue
            candidates: list[Any] = [result.get("video")]
            videos = result.get("videos")
            if isinstance(videos, list):
                candidates.extend(videos)
            # Some video-producing integrations return the field at the top level.
            candidates.append(result)
            for value in candidates:
                if not isinstance(value, dict):
                    continue
                name = value.get("video_name")
                if isinstance(name, str) and name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def get_video_metadata(self, video_name: str) -> str | None:
        """Fetch the remote video record metadata before the native local import."""
        encoded_name = urllib.parse.quote(video_name, safe="")
        value = self._request_json_value("GET", f"/api/v1/videos/i/{encoded_name}/metadata")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RemoteInvokeError(f"Remote video '{video_name}' returned invalid metadata")
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    def get_video_dto(self, video_name: str) -> dict[str, Any]:
        encoded_name = urllib.parse.quote(video_name, safe="")
        return self._request_json("GET", f"/api/v1/videos/i/{encoded_name}")

    def filter_gallery_video_names(self, video_names: list[str]) -> list[str]:
        """Use the remote's actual persisted Save-to-Gallery flag, as for images."""
        gallery_names: list[str] = []
        for video_name in video_names:
            if self.get_video_dto(video_name).get("is_intermediate") is False:
                gallery_names.append(video_name)
        return gallery_names

    def download_video(self, video_name: str) -> bytes:
        encoded_name = urllib.parse.quote(video_name, safe="")
        return self._request("GET", f"/api/v1/videos/i/{encoded_name}/full")

    def delete_video(self, video_name: str) -> dict[str, Any]:
        encoded_name = urllib.parse.quote(video_name, safe="")
        return self._request_json("DELETE", f"/api/v1/videos/i/{encoded_name}")

    def upload_input_image(self, image: Image.Image) -> str:
        """Upload a local input as an InvokeAI-owned remote image and return its NEW name.

        Intermediate+user category prevents source inputs appearing as finished Gallery
        results. This uses the same authenticated client (including the 401 retry).
        """
        png = BytesIO()
        try:
            image.save(png, format="PNG")
        except Exception as exc:
            raise RemoteInvokeError(f"Could not encode source image as PNG: {exc}") from exc
        boundary = f"irw-{secrets.token_hex(16)}"
        prefix = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="remote-input.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8")
        body = prefix + png.getvalue() + f"\r\n--{boundary}--\r\n".encode("utf-8")
        query = urllib.parse.urlencode({"image_category": "user", "is_intermediate": "true"})
        path = f"/api/v1/images/upload?{query}"
        raw = self._request("POST", path, body=body, content_type=f"multipart/form-data; boundary={boundary}")
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise RemoteInvokeError("Remote image upload did not return valid JSON") from exc
        remote_name = record.get("image_name") if isinstance(record, dict) else None
        if not isinstance(remote_name, str) or not remote_name:
            raise RemoteInvokeError("Remote image upload did not return image_name")
        return remote_name

    def upload_input_video(self, video_path: Path) -> str:
        """Stream a local source video to the remote as an intermediate input."""
        path = Path(video_path)
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise RemoteInvokeError(f"Could not read source video '{path}': {exc}") from exc

        boundary = f"irw-{secrets.token_hex(16)}"
        prefix = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="remote-input.mp4"\r\n'
            "Content-Type: video/mp4\r\n\r\n"
        ).encode("utf-8")
        suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")

        class MultipartVideoBody:
            def __iter__(self):
                yield prefix
                try:
                    with path.open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            yield chunk
                except OSError as exc:
                    raise RemoteInvokeError(f"Could not read source video '{path}': {exc}") from exc
                yield suffix

        query = urllib.parse.urlencode({"video_category": "user", "is_intermediate": "true"})
        raw = self._request(
            "POST",
            f"/api/v1/videos/upload?{query}",
            body=MultipartVideoBody(),
            content_type=f"multipart/form-data; boundary={boundary}",
            content_length=len(prefix) + file_size + len(suffix),
        )
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise RemoteInvokeError("Remote video upload did not return valid JSON") from exc
        remote_name = record.get("video_name") if isinstance(record, dict) else None
        if not isinstance(remote_name, str) or not remote_name:
            raise RemoteInvokeError("Remote video upload did not return video_name")
        return remote_name

    def get_image_metadata(self, image_name: str) -> str | None:
        """Fetch the exact image record metadata, not PNG/PIL metadata."""
        encoded_name = urllib.parse.quote(image_name, safe="")
        value = self._request_json_value("GET", f"/api/v1/images/i/{encoded_name}/metadata")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RemoteInvokeError(f"Remote image '{image_name}' returned invalid metadata")
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    def get_image_dto(self, image_name: str) -> dict[str, Any]:
        encoded_name = urllib.parse.quote(image_name, safe="")
        return self._request_json("GET", f"/api/v1/images/i/{encoded_name}")

    def filter_gallery_image_names(self, image_names: list[str]) -> list[str]:
        """Return only images that InvokeAI itself marks as non-intermediate.

        InvokeAI's Save to Gallery state is persisted on the generated image record as
        ``is_intermediate``. Filtering the completed session's image names through the
        remote ImageDTO avoids relying on graph/result node-id correspondence, which can
        differ after graph expansion and execution.
        """
        gallery_names: list[str] = []
        for image_name in image_names:
            dto = self.get_image_dto(image_name)
            if dto.get("is_intermediate") is False:
                gallery_names.append(image_name)
        return gallery_names

    def delete_image(self, image_name: str) -> dict[str, Any]:
        """Delete an image through InvokeAI's normal image service API.

        This removes the image record and lets InvokeAI clean up its associated
        stored/thumbnail/cache files through the same path the UI uses.
        """
        encoded_name = urllib.parse.quote(image_name, safe="")
        return self._request_json("DELETE", f"/api/v1/images/i/{encoded_name}")

    def download_image(self, image_name: str) -> Image.Image:
        encoded_name = urllib.parse.quote(image_name, safe="")
        raw = self._request("GET", f"/api/v1/images/i/{encoded_name}/full")
        try:
            with Image.open(BytesIO(raw)) as image:
                image.load()
                return image.copy()
        except Exception as exc:
            raise RemoteInvokeError(f"Downloaded remote image '{image_name}' could not be decoded") from exc
