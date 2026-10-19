# Gallery listings

Paths are relative to the webv2 `src/` directory.

Gallery ordinary listings are Query-owned sparse pages shared by grid, picker, Preview, and deep-reveal flows. The Gallery widget's media kind (both, images, or videos) is part of the canonical listing filter, present only when restricted; the picker always lists both kinds. Page key = account lifetime + canonical listing filters + aligned absolute offset; each result keeps its authoritative total and absolute item positions so hydrated gaps never compact. Consumers own separate visible ranges and scroll anchors. Cold numbered page after page zero first requests count-only metadata, then reads only its clamped 60-item page; virtual scrolling discovers count from its first item page. Recent local completions use a temporary overlay, with listing-scoped membership evidence bounded by recent-item limit so an authoritative item does not reappear in overlay when its page is evicted. Query layer retains at most ten inactive idle pages per listing while protecting observed pages, and schedules at most four distinct page reads per QueryClient. Deep reveal resolves an item's filtered listing rank, reads its containing page, verifies item at returned absolute position, and retries that rank/page check at most once if offsets shifted. Persisted Preview selection keeps listing identity, media kind included, from when item was selected (a stamp without a media kind lists both); changing active Gallery filters does not rewrite that cursor. Preview re-resolves a stale item page after listing mutations; its async navigation is fenced by captured account epoch so a late page cannot stamp selection into a newer account lifetime.

## Realtime refresh and external reveal

Uploads, reconnects, and the signed-in user's completed queue items whose origin does not start with `webv2:` refresh the listings. This client's own items are refreshed by Queue, in the tab that tracks them only.

An external client asks for its results to be revealed by enqueueing with `POST /api/v1/queue/default/enqueue_batch` and `batch.destination` set to `gallery:reveal`. The request is honored only for the signed-in user's items (any user's on a single-user server) whose origin does not start with `webv2:`. Every other destination only refreshes, `gallery` included: legacy web sends it on every workflow and upscale run.

The reveal is best effort. Each completed item asks once; a reveal that stands down is dropped, not retried, except in a hidden tab. Every visible tab or device signed in as that user may reveal. The reveal takes the last non-intermediate image the item saved and lands on it as Find in Gallery does: board, page and selection, plus the starred-only filter, the media filter and the images/assets view when the item needs them, after which Preview follows the selection and the grid scrolls to it. With the Gallery widget closed, the selection still changes, unseen. It raises no widget and leaves Canvas, Generate panels and the route untouched. Only while the Gallery widget is mounted is it announced, once, in the Gallery's polite live region; a Gallery mounted later does not repeat it.

It never takes the Gallery from the user. These conditions are checked when the completion arrives, before and after its result is read, and immediately before the first Gallery change; the first failure drops the reveal:

- the user is not typing (this window has focus and a text field in it does);
- the project on screen at arrival is still the active one and has no open run of its own;
- no search text, semantic text or semantic query narrows the listing (an open but empty semantic field does not count);
- at most one item is selected;
- no other reveal, such as Find in Gallery or an image-map click, is locating its item;
- the user has not navigated the Gallery since arrival: board, selection, view, starred-only, media filter, search, page, or a claimed navigation;
- no newer reveal request has arrived. Only the newest completion reveals, so a batch of several runs reveals at most its last completed item.

A hidden tab defers instead: the newest request waits until the tab is shown, and the conditions are checked then.

Known limitations:

- The status event does not say whether an item is a workflow-call child, which inherits its parent's destination. A child's completion supersedes a reveal still in flight before its read finds it is a child and reveals nothing; if the parent then fails or is cancelled, the earlier item is not revealed.
- An image a graph passes through from its inputs counts as an output, so it is picked if it is the last non-intermediate one.

## Board destinations

Where results and uploads land is decided from a cached board listing, never guessed: `canAddToCachedGalleryBoard` (`features/gallery/data/queryCache.ts`) takes a board as a destination only when a cached listing holds it with `add_media`, archived or not; a board no listing holds is not one. Submissions and uploads call `ensureGalleryBoardDestinations` first. It resolves optimistically: any cached listing of every readable board (archived included) decides at once, whatever its age; one older than a minute or invalidated is read again in the background for the next resolution; only with no such listing cached is one read awaited, and if that read fails nothing is a destination and results go to the inbox. No surface observes that listing just in case: the Gallery panel's own listing serves when it shows archived boards, and otherwise the first submission or upload reads one. A failed gallery-bound run, workflow run or upload, including one a board refused with 403 or answered 404 for, calls `invalidateGalleryBoardDestinations`: the gallery is invalidated and the listing of every readable board is read again at once, with no observer needed. A submission or upload made after that read lands decides from the new listing; one made before it lands still decides from the old listing and shares the read. The gallery upload action refuses a real board the ensured listing does not take uploads for before sending anything, and says whether the board is view-only or gone. Graph compilation applies the same predicate to explicit workflow board fields; a refused board resolves like Auto. A board the server answers 403 for exists but is unreadable: it stays named by the project and is shown as the project board meanwhile; only a 404 (deleted) reconciles the project's references. Hosts without board sharing fold every board into the caller's own through `useHostGalleryBoards`, which the panel, board menus and workflow board fields share.
