from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from invokeai.app.services.intermediates.intermediates_common import (
    IntermediatesBrowserHoldRequest,
    IntermediatesOperation,
    IntermediatesOperationRequest,
    IntermediatesPreview,
    IntermediatesPreviewRequest,
    IntermediatesSummary,
    IntermediatesSummarySort,
)
from invokeai.app.services.shared.media_references import MediaReferences


@dataclass(frozen=True)
class IntermediatesCaller:
    """Who is asking: the authenticated account and whether it administers the instance."""

    user_id: str
    is_admin: bool


class IntermediatesServiceBase(ABC):
    """Scoped cleanup of intermediate images and videos under one eligibility policy."""

    @abstractmethod
    def hold_cached_media(self, session_id: str, references: MediaReferences) -> bool:
        """Pin existing cached media until the consuming queue session ends; False means recompute."""
        pass

    @abstractmethod
    def replace_browser_hold(
        self, caller: IntermediatesCaller, lease_id: str, request: IntermediatesBrowserHoldRequest
    ) -> None:
        pass

    @abstractmethod
    def release_browser_hold(self, caller: IntermediatesCaller, lease_id: str) -> None:
        pass

    @abstractmethod
    def get_summary(
        self,
        caller: IntermediatesCaller,
        *,
        owner_id: Optional[str],
        search: Optional[str],
        sort: IntermediatesSummarySort,
        descending: bool,
        offset: int,
        limit: int,
        project_id: Optional[str] = None,
    ) -> IntermediatesSummary:
        pass

    @abstractmethod
    def create_preview(self, request: IntermediatesPreviewRequest, caller: IntermediatesCaller) -> IntermediatesPreview:
        pass

    @abstractmethod
    def start_operation(
        self, request: IntermediatesOperationRequest, caller: IntermediatesCaller
    ) -> IntermediatesOperation:
        pass

    @abstractmethod
    def get_operation(self, operation_id: str, caller: IntermediatesCaller) -> IntermediatesOperation:
        pass

    @abstractmethod
    def list_operations(self, caller: IntermediatesCaller) -> list[IntermediatesOperation]:
        """The caller's retained operations, newest first."""
        pass

    @abstractmethod
    def clear_all_images_now(self, caller: IntermediatesCaller) -> int:
        """The legacy instance-wide clear: every safe intermediate image, synchronously. Admin only."""
        pass

    @abstractmethod
    def count_safe_images(self, caller: IntermediatesCaller) -> int:
        """The legacy count: the intermediate images `clear_all_images_now` would delete, within the caller's view."""
        pass
