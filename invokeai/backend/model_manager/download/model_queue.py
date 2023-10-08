import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

from huggingface_hub import HfApi, hf_hub_url
from pydantic import BaseModel, Field, parse_obj_as, validator
from pydantic.networks import AnyHttpUrl

from .base import DownloadEventHandler, DownloadJobBase, DownloadJobStatus, DownloadQueueBase
from .queue import HTTP_RE, DownloadJobRemoteSource, DownloadQueue

# regular expressions used to dispatch appropriate downloaders and metadata retrievers
# endpoint for civitai get-model API
CIVITAI_MODEL_DOWNLOAD = r"https://civitai.com/api/download/models/(\d+)"
CIVITAI_MODEL_PAGE = "https://civitai.com/models/"
CIVITAI_MODEL_PAGE_WITH_VERSION = r"https://civitai.com/models/(\d+)\?modelVersionId=(\d+)"
CIVITAI_MODELS_ENDPOINT = "https://civitai.com/api/v1/models/"
CIVITAI_VERSIONS_ENDPOINT = "https://civitai.com/api/v1/model-versions/"

# Regular expressions to describe repo_ids and http urls
REPO_ID_WITH_OPTIONAL_SUBFOLDER_RE = r"^([.\w-]+/[.\w-]+)(?::([.\w-]+))?$"


class ModelSourceMetadata(BaseModel):
    """Information collected on a downloadable model from its source site."""

    name: Optional[str] = Field(description="Human-readable name of this model")
    author: Optional[str] = Field(description="Author/creator of the model")
    description: Optional[str] = Field(description="Description of the model")
    license: Optional[str] = Field(description="Model license terms")
    thumbnail_url: Optional[AnyHttpUrl] = Field(description="URL of a thumbnail image for the model")
    tags: Optional[List[str]] = Field(description="List of descriptive tags")


class DownloadJobWithMetadata(DownloadJobRemoteSource):
    """A remote download that has metadata associated with it."""

    metadata: ModelSourceMetadata = Field(
        description="Metadata describing the model, derived from source", default_factory=ModelSourceMetadata
    )


class DownloadJobRepoID(DownloadJobWithMetadata):
    """Download repo ids."""

    source: str = Field(description="A repo_id (foo/bar), or a repo_id with a subfolder (foo/far:v2)")
    subfolder: Optional[str] = Field(
        description="Provide when the desired model is in a subfolder of the repo_id's distro", default=None
    )
    variant: Optional[str] = Field(description="Variant, such as 'fp16', to download")
    subqueue: Optional[DownloadQueueBase] = Field(
        description="a subqueue used for downloading the individual files in the repo_id", default=None
    )

    @validator("source")
    @classmethod
    def proper_repo_id(cls, v: str) -> str:  # noqa D102
        if not re.match(REPO_ID_WITH_OPTIONAL_SUBFOLDER_RE, v):
            raise ValueError(f"{v}: invalid repo_id format")
        return v

    def cleanup(self, preserve_partial_downloads: bool = False):
        """Perform action when job is completed."""
        if self.subqueue:
            self.subqueue.cancel_all_jobs(preserve_partial=preserve_partial_downloads)
            self.subqueue.release()


class ModelDownloadQueue(DownloadQueue):
    """Subclass of DownloadQueue, able to retrieve metadata from HuggingFace and Civitai."""

    def create_download_job(
        self,
        source: Union[str, Path, AnyHttpUrl],
        destdir: Path,
        start: bool = True,
        priority: int = 10,
        filename: Optional[Path] = None,
        variant: Optional[str] = None,
        access_token: Optional[str] = None,
        event_handlers: List[DownloadEventHandler] = [],
    ) -> DownloadJobBase:
        """Create a download job and return its ID."""
        cls: Optional[Type[DownloadJobBase]] = None
        kwargs: Dict[str, Optional[str]] = dict()

        if re.match(HTTP_RE, str(source)):
            cls = DownloadJobWithMetadata
            kwargs.update(access_token=access_token)
        elif re.match(REPO_ID_WITH_OPTIONAL_SUBFOLDER_RE, str(source)):
            cls = DownloadJobRepoID
            kwargs.update(
                variant=variant,
                access_token=access_token,
            )
        if cls:
            job = cls(
                source=source,
                destination=Path(destdir) / (filename or "."),
                event_handlers=event_handlers,
                priority=priority,
                **kwargs,
            )
            return self.submit_download_job(job, start)
        else:
            return super().create_download_job(
                source=source,
                destdir=destdir,
                start=start,
                priority=priority,
                filename=filename,
                variant=variant,
                access_token=access_token,
                event_handlers=event_handlers,
            )

    def select_downloader(self, job: DownloadJobBase) -> Callable[[DownloadJobBase], None]:
        """Based on the job type select the download method."""
        if isinstance(job, DownloadJobRepoID):
            return self._download_repoid
        elif isinstance(job, DownloadJobWithMetadata):
            return self._download_with_resume
        else:
            return super().select_downloader(job)

    def get_url_for_job(self, job: DownloadJobBase) -> AnyHttpUrl:
        """
        Fetch metadata from certain well-known URLs.

        The metadata will be stashed in job.metadata, if found
        Return the download URL.
        """
        assert isinstance(job, DownloadJobWithMetadata)
        metadata = job.metadata
        url = job.source
        metadata_url = url
        model = None

        # a Civitai download URL
        if match := re.match(CIVITAI_MODEL_DOWNLOAD, str(metadata_url)):
            version = match.group(1)
            resp = self._requests.get(CIVITAI_VERSIONS_ENDPOINT + version).json()
            metadata.thumbnail_url = metadata.thumbnail_url or resp["images"][0]["url"]
            metadata.description = metadata.description or (
                f"Trigger terms: {(', ').join(resp['trainedWords'])}" if resp["trainedWords"] else resp["description"]
            )
            metadata_url = CIVITAI_MODEL_PAGE + str(resp["modelId"]) + f"?modelVersionId={version}"

        # a Civitai model page with the version
        if match := re.match(CIVITAI_MODEL_PAGE_WITH_VERSION, str(metadata_url)):
            model = match.group(1)
            version = int(match.group(2))
        # and without
        elif match := re.match(CIVITAI_MODEL_PAGE + r"(\d+)", str(metadata_url)):
            model = match.group(1)
            version = None

        if not model:
            return parse_obj_as(AnyHttpUrl, url)

        if model:
            resp = self._requests.get(CIVITAI_MODELS_ENDPOINT + str(model)).json()

            metadata.author = metadata.author or resp["creator"]["username"]
            metadata.tags = metadata.tags or resp["tags"]
            metadata.license = (
                metadata.license
                or f"allowCommercialUse={resp['allowCommercialUse']}; allowDerivatives={resp['allowDerivatives']}; allowNoCredit={resp['allowNoCredit']}"
            )

            if version:
                versions = [x for x in resp["modelVersions"] if int(x["id"]) == version]
                version_data = versions[0]
            else:
                version_data = resp["modelVersions"][0]  # first one

            metadata.thumbnail_url = version_data.get("url") or metadata.thumbnail_url
            metadata.description = metadata.description or (
                f"Trigger terms: {(', ').join(version_data.get('trainedWords'))}"
                if version_data.get("trainedWords")
                else version_data.get("description")
            )

            download_url = version_data["downloadUrl"]

        # return the download url
        return download_url

    def _download_repoid(self, job: DownloadJobBase) -> None:
        """Download a job that holds a huggingface repoid."""

        def subdownload_event(subjob: DownloadJobBase):
            assert isinstance(subjob, DownloadJobRemoteSource)
            assert isinstance(job, DownloadJobRemoteSource)
            if subjob.status == DownloadJobStatus.RUNNING:
                bytes_downloaded[subjob.id] = subjob.bytes
                job.bytes = sum(bytes_downloaded.values())
                self._update_job_status(job, DownloadJobStatus.RUNNING)
                return

            if subjob.status == DownloadJobStatus.ERROR:
                job.error = subjob.error
                job.cleanup()
                self._update_job_status(job, DownloadJobStatus.ERROR)
                return

            if subjob.status == DownloadJobStatus.COMPLETED:
                bytes_downloaded[subjob.id] = subjob.bytes
                job.bytes = sum(bytes_downloaded.values())
                self._update_job_status(job, DownloadJobStatus.RUNNING)
                return

        subqueue = self.__class__(
            event_handlers=[subdownload_event],
            requests_session=self._requests,
            quiet=True,
        )
        assert isinstance(job, DownloadJobRepoID)
        try:
            repo_id = job.source
            variant = job.variant
            if not job.metadata:
                job.metadata = ModelSourceMetadata()
            urls_to_download = self._get_repo_info(
                repo_id, variant=variant, metadata=job.metadata, subfolder=job.subfolder
            )
            if job.destination.name != Path(repo_id).name:
                job.destination = job.destination / Path(repo_id).name
            bytes_downloaded: Dict[int, int] = dict()
            job.total_bytes = 0

            for url, subdir, file, size in urls_to_download:
                job.total_bytes += size
                subqueue.create_download_job(
                    source=url,
                    destdir=job.destination / subdir,
                    filename=file,
                    variant=variant,
                    access_token=job.access_token,
                )
        except KeyboardInterrupt as excp:
            raise excp
        except Exception as excp:
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)
            self._logger.error(job.error)
        finally:
            job.subqueue = subqueue
            job.subqueue.join()
            if job.status == DownloadJobStatus.RUNNING:
                self._update_job_status(job, DownloadJobStatus.COMPLETED)
            job.subqueue.release()  # get rid of the subqueue

    def _get_repo_info(
        self,
        repo_id: str,
        metadata: ModelSourceMetadata,
        variant: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> List[Tuple[AnyHttpUrl, Path, Path, int]]:
        """
        Given a repo_id and an optional variant, return list of URLs to download to get the model.

        The metadata field will be updated with model metadata from HuggingFace.

        Known variants currently are:
        1. onnx
        2. openvino
        3. fp16
        4. None (usually returns fp32 model)
        """
        model_info = HfApi().model_info(repo_id=repo_id, files_metadata=True)
        sibs = model_info.siblings
        paths = [x.rfilename for x in sibs]
        sizes = {x.rfilename: x.size for x in sibs}

        prefix = ""
        if subfolder:
            prefix = f"{subfolder}/"
            paths = [x for x in paths if x.startswith(prefix)]

        if f"{prefix}model_index.json" in paths:
            url = hf_hub_url(repo_id, filename="model_index.json", subfolder=subfolder)
            resp = self._requests.get(url)
            resp.raise_for_status()  # will raise an HTTPError on non-200 status
            submodels = resp.json()
            paths = [Path(subfolder or "", x) for x in paths if Path(x).parent.as_posix() in submodels]
            paths.insert(0, f"{prefix}model_index.json")
        urls = [
            (
                hf_hub_url(repo_id, filename=x.as_posix()),
                x.parent.relative_to(prefix) or Path("."),
                Path(x.name),
                sizes[x.as_posix()],
            )
            for x in self._select_variants(paths, variant)
        ]
        if hasattr(model_info, "cardData"):
            metadata.license = metadata.license or model_info.cardData.get("license")
        metadata.tags = metadata.tags or model_info.tags
        metadata.author = metadata.author or model_info.author
        return urls

    def _select_variants(self, paths: List[str], variant: Optional[str] = None) -> Set[Path]:
        """Select the proper variant files from a list of HuggingFace repo_id paths."""
        result = set()
        basenames: Dict[Path, Path] = dict()
        for p in paths:
            path = Path(p)

            if path.suffix == ".onnx":
                if variant == "onnx":
                    result.add(path)

            elif path.name.startswith("openvino_model"):
                if variant == "openvino":
                    result.add(path)

            elif path.suffix in [".json", ".txt"]:
                result.add(path)

            elif path.suffix in [".bin", ".safetensors", ".pt"] and variant in ["fp16", None]:
                parent = path.parent
                suffixes = path.suffixes
                if len(suffixes) == 2:
                    file_variant, suffix = suffixes
                    basename = parent / Path(path.stem).stem
                else:
                    file_variant = None
                    suffix = suffixes[0]
                    basename = parent / path.stem

                if previous := basenames.get(basename):
                    if previous.suffix != ".safetensors" and suffix == ".safetensors":
                        basenames[basename] = path
                    if file_variant == f".{variant}":
                        basenames[basename] = path
                    elif not variant and not file_variant:
                        basenames[basename] = path
                else:
                    basenames[basename] = path

            else:
                continue

        for v in basenames.values():
            result.add(v)

        return result
