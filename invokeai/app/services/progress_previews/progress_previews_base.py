from abc import ABC, abstractmethod

from invokeai.app.services.events.events_common import InvocationProgressEvent


class ProgressPreviewsBase(ABC):
    """The latest denoising preview per running queue item: best-effort and disposable.

    Progress frames travel over the socket as base64 data URLs and otherwise exist nowhere. A
    client that reconnects, or a tab that was hidden while its socket was dropped, needs to ask for
    the last frame rather than wait for the next step. This store keeps the last *emitted* progress
    event per queue item so it can be replayed on subscribe or fetched over HTTP.

    Each frame carries a `revision` that is monotonic per queue item and session; the client drops
    any frame at or below the revision it has already shown, so a replay can never move the preview
    backwards. A process restart empties the store, which is the intended epoch semantics.

    Image-bearing frames are also throttled here (`reserve_revision` returns None for a frame that
    should be dropped) so that a fast GPU does not turn every denoising step into a JPEG encode and
    a socket emit.
    """

    @abstractmethod
    def reserve_revision(self, item_id: int, session_id: str, percentage: float | None) -> int | None:
        """Allocate the revision for a new image-bearing frame, or None if the frame should be dropped.

        Args:
            item_id: The queue item the frame belongs to.
            session_id: The session producing it; a new session on the same item starts a new epoch.
            percentage: The frame's progress, if determinate. A final frame (1.0) is never dropped.
        """
        pass

    @abstractmethod
    def record(self, event: InvocationProgressEvent) -> None:
        """Remember an emitted image-bearing progress event as the item's latest preview."""
        pass

    @abstractmethod
    def get(self, item_id: int) -> InvocationProgressEvent | None:
        """The latest preview for a queue item, if any."""
        pass

    @abstractmethod
    def list_for_user(self, user_id: str, queue_id: str | None = None) -> list[InvocationProgressEvent]:
        """The latest previews for every running queue item the user owns, oldest item first."""
        pass

    @abstractmethod
    def clear_node(self, item_id: int, invocation_id: str) -> None:
        """Forget the preview if this node produced it: the node is done, so a replay of its frame
        would show a denoise that has already finished and mark the node as running again."""
        pass

    @abstractmethod
    def clear(self, item_id: int) -> None:
        """Forget a queue item's preview: its session leg ended, whatever the outcome. The item's
        revision counter is kept so a later leg of the same session cannot reissue a revision a
        client has already seen."""
        pass

    @abstractmethod
    def clear_all(self) -> None:
        pass
