import type { Dispatch, UnknownAction } from '@reduxjs/toolkit';
import type { FetchArgs, FetchBaseQueryError, QueryReturnValue } from '@reduxjs/toolkit/query';
import { skipToken } from '@reduxjs/toolkit/query';
import { getStore } from 'app/store/nanostores/store';
import { uniq } from 'es-toolkit';
import type { AuthContext } from 'features/auth/store/authTokenRefresh';
import { captureAuthContext, isSameAuthContext } from 'features/auth/store/authTokenRefresh';
import type { CroppableImageWithDims } from 'features/controlLayers/store/types';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { toast } from 'features/toast/toast';
import i18n from 'i18next';
import type { components, paths } from 'services/api/schema';
import type {
  GetImageNamesArgs,
  GetImageNamesResult,
  GraphAndWorkflowResponse,
  ImageDTO,
  ImageUploadEntryRequest,
  ImageUploadEntryResponse,
  ListImagesArgs,
  ListImagesResponse,
  UploadImageArg,
} from 'services/api/types';
import { getListImagesUrl } from 'services/api/util';
import {
  getTagsToInvalidateForBoardAffectingMutation,
  getTagsToInvalidateForImageMutation,
  getTagsToInvalidateForVideoMutation,
} from 'services/api/util/tagInvalidation';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';
import type { JsonObject } from 'type-fest';

import { api, buildV1Url, LIST_TAG } from '..';
import { buildBoardsUrl } from './boards';

/**
 * Builds an endpoint URL for the images router
 * @example
 * buildImagesUrl('some-path')
 * // '/api/v1/images/some-path'
 */
const buildImagesUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`images/${path}`, query);

/**
 * Builds an endpoint URL for the board_images router
 * @example
 * buildBoardImagesUrl('some-path')
 * // '/api/v1/board_images/some-path'
 */
const buildBoardImagesUrl = (path: string = '') => buildV1Url(`board_images/${path}`);

/**
 * The batch routes cap `image_names` server-side (`MAX_IMAGE_BATCH_SIZE` in
 * `invokeai/app/api/routers/images.py`) so one authenticated client cannot pin a worker with a
 * single request. Nothing caps a gallery *selection*: select-all reads the whole board's name
 * list, so one keystroke on a large board produces a selection an order of magnitude past the
 * cap. Oversized bodies are therefore split client-side into conforming requests.
 *
 * Must stay <= the backend constant.
 */
const IMAGE_BATCH_CHUNK_SIZE = 1000;

/**
 * Every batch route answers with an object whose values are all name lists, and every one of
 * them reports its per-name failures in `failed_images` — which is where a chunked run folds
 * the names it never reached.
 */
type ImageBatchResult = Record<string, string[]> & { failed_images: string[] };

type InvalidateTagsArg = Parameters<typeof api.util.invalidateTags>[0];

/** The `baseQuery` handed to a `queryFn`, matching what `fetchBaseQuery` produces. */
type ImagesBaseQuery = (
  args: string | FetchArgs
) => QueryReturnValue<unknown, FetchBaseQueryError> | PromiseLike<QueryReturnValue<unknown, FetchBaseQueryError>>;

/**
 * Another user's session now owns the tab. This is the hard abort: nothing from the old run may
 * be consumed — not a payload, not a partial aggregate, not a toast — because everything it
 * carries belongs to whoever started it, and the loops treat it as fatal (`isAuthChangedError`).
 */
const AUTH_CHANGED_ERROR: FetchBaseQueryError = {
  status: 'CUSTOM_ERROR',
  error: 'Aborted: the authenticated session changed while the operation was running',
};

/**
 * The session ended with no successor: the token is gone — a 401 on *any* concurrent request
 * dispatches `sessionExpiredLogout`, whose reducer clears `auth_token` synchronously, and a
 * deliberate logout does the same — and nobody else has logged in. Deliberately NOT matched by
 * `isAuthChangedError`, so it takes the same partial-success path as an ordinary chunk failure:
 * there is no new session to protect, the committed chunks belong to the very user who is about
 * to land on the login screen, and the partial reporting is what lets `handleDeletions` strip
 * committed deletions out of the persisted slices (canvas, nodes, reference images — none of
 * which handle `sessionExpiredLogout`) before their next login.
 */
const SESSION_ENDED_ERROR: FetchBaseQueryError = {
  status: 'CUSTOM_ERROR',
  error: 'Aborted: the authenticated session ended while the operation was running',
};

const isAuthChangedError = (error: FetchBaseQueryError | undefined): boolean =>
  error?.status === AUTH_CHANGED_ERROR.status && error.error === AUTH_CHANGED_ERROR.error;

/** Either flavor of session mismatch — the "consume nothing from this run" superset. */
const isSessionMismatchError = (error: FetchBaseQueryError | undefined): boolean =>
  isAuthChangedError(error) ||
  (error?.status === SESSION_ENDED_ERROR.status && error.error === SESSION_ENDED_ERROR.error);

/**
 * Errors that leave a chunk's outcome unknown. A transport failure or timeout can strike after
 * the request reached the server -- fetch loses the response, not necessarily the request -- a
 * parsing error means a response arrived for work that was already done, and a 5xx says the
 * route died somewhere in a loop whose per-name writes had each already committed. In all of
 * these the server may have applied the chunk with only the report lost. A 4xx is the server
 * itself saying it refused, which is the one shape that proves the chunk did nothing.
 */
const isIndeterminateError = (error: FetchBaseQueryError): boolean =>
  error.status === 'FETCH_ERROR' ||
  error.status === 'TIMEOUT_ERROR' ||
  error.status === 'PARSING_ERROR' ||
  (typeof error.status === 'number' && error.status >= 500);

/**
 * Which of the two a failed session check means. The distinction is the whole ballgame: expiry
 * must degrade into an ordinary failure so committed work is still reported, while a takeover
 * must abort hard so the new user consumes nothing. Collapsing them in either direction is a
 * bug this file has now had both ways.
 *
 * Exported for its unit test, and the test exists because no loop-level test can reach the
 * takeover arm at the pre-request check: same-tab, everything from one chunk's post-response
 * check to the next chunk's pre-request check is a single synchronous drain no dispatch can
 * interleave with, so a takeover staged inside a mocked baseQuery is always caught by the
 * post-response check first. The pre-request arm is live only cross-tab — another tab's
 * login-as-B writes localStorage between chunks — which is real concurrency a same-thread test
 * cannot stage, and it is the guard that stops the next chunk going out under B's token.
 */
export const sessionMismatchError = (): FetchBaseQueryError =>
  localStorage.getItem('auth_token') === null ? SESSION_ENDED_ERROR : AUTH_CHANGED_ERROR;

/**
 * Issues one chunk of a multi-request operation, unless the session it started under is gone.
 *
 * A selection past the batch cap is split into several requests, and each one picks up whatever
 * bearer token localStorage holds when it is sent — the loop does not carry the identity it
 * started with. Log out and back in as another user with the loop still running and the
 * remaining chunks are applied as that user; on a public board they can land, so half of one
 * user's delete is committed under another's name, with nothing to roll it back. Nothing else
 * stops it: `resetApiState` on the logout action clears the cache, not a running `queryFn`.
 *
 * So the context is captured before the first chunk, rechecked before each request, and checked
 * again before its response is consumed. See `isSameAuthContext` for what counts as the same
 * session — notably a sliding-window token refresh does not, or every long batch would abandon
 * itself. A failed check is then triaged by `sessionMismatchError`, because the two ways it can
 * fail call for opposite treatments — and the token can vanish at *any* await in the run, not
 * just this chunk's own request: a 401 on any concurrent request (a gallery poll, a board
 * refetch) clears it synchronously mid-flight. Collapse expiry into the hard abort and every
 * one of those windows silently discards whatever the earlier chunks already committed.
 */
const fetchChunk = async (
  baseQuery: ImagesBaseQuery,
  authContext: AuthContext,
  args: FetchArgs
): Promise<QueryReturnValue<unknown, FetchBaseQueryError>> => {
  if (!isSameAuthContext(authContext)) {
    return { error: sessionMismatchError() };
  }
  const response = await baseQuery(args);
  // An error passes through untriaged: it carries nothing the next session could consume, and
  // rewriting it loses what it was. This chunk's own expired-session 401 is the everyday case —
  // `dynamicBaseQuery` dispatches `sessionExpiredLogout` before returning it, so the token is
  // already gone by this line, and a rewrite would turn the ordinary failure into an abort.
  if (response.error) {
    return response;
  }
  // A *successful* payload is consumed unless another user has taken over. Mere expiry keeps
  // the payload: it belongs to the same user, the server committed it, and dropping it would
  // un-report work that already happened — the loop then stops at the next pre-request check.
  const token = localStorage.getItem('auth_token');
  if (token !== null && !isSameAuthContext(authContext)) {
    return { error: AUTH_CHANGED_ERROR };
  }
  return response;
};

export const chunkImageNames = (image_names: string[]): string[][] => {
  if (image_names.length <= IMAGE_BATCH_CHUNK_SIZE) {
    // Single-request path, byte-for-byte what it was before chunking existed. Note this also
    // covers the empty list: the routes answer it with a well-formed empty result, which
    // merging zero chunks could not reproduce.
    return [image_names];
  }
  const chunks: string[][] = [];
  for (let i = 0; i < image_names.length; i += IMAGE_BATCH_CHUNK_SIZE) {
    chunks.push(image_names.slice(i, i + IMAGE_BATCH_CHUNK_SIZE));
  }
  return chunks;
};

/**
 * Unions the per-chunk results key-by-key, so callers and `invalidatesTags` see one aggregate
 * result with the same shape a single request would have returned. Keys are unioned rather
 * than enumerated because the five batch routes name their outcome list differently
 * (`deleted_images`, `starred_images`, `added_images`, ...) while sharing `affected_boards`.
 */
export const mergeImageBatchResults = <TResult extends ImageBatchResult>(results: TResult[]): TResult => {
  // Accumulator is the looser Record: it is only complete once every chunk has contributed.
  const merged: Record<string, string[]> = {};
  for (const result of results) {
    for (const [key, names] of Object.entries(result)) {
      merged[key] = merged[key] ? uniq(merged[key].concat(names)) : names;
    }
  }
  return merged as TResult;
};

/**
 * Builds a `queryFn` that runs a batch mutation one conforming chunk at a time.
 *
 * Chunks go sequentially on purpose. Each is already up to `IMAGE_BATCH_CHUNK_SIZE` rows of DB
 * work, and the server-side bound exists to stop a single client from pinning a worker — firing
 * the chunks concurrently would hand straight back what the bound took away.
 *
 * The mid-run failure is the case that matters, and it resolves to a partial success rather than
 * an error. Earlier chunks have already been committed by the server; returning a bare error
 * would discard their payload, which is exactly the bug the partial-failure reporting on these
 * routes exists to fix. It is not only the RTK cache at stake — `handleDeletions` drives the
 * gallery selection and strips deleted images out of nodes, canvas layers and reference images
 * off `result.deleted_images`, and none of that runs on a rejection. So the merged result is
 * returned, the unreached names are folded into its `failed_images`, and only a run where
 * *nothing* landed surfaces as an error.
 *
 * They are folded into the payload rather than toasted here so that a single `onQueryStarted`
 * reports one total. Toasting from both places would fire twice on the same toast id, and the
 * toast system updates in place and appends "(2)" — so the second count would simply replace
 * the first rather than adding to it.
 */
export const buildChunkedImageBatchQueryFn =
  <TResult extends ImageBatchResult, TArg extends { image_names: string[] }>(
    request: (body: TArg) => { url: string; method: string },
    getTags: (result: TResult) => InvalidateTagsArg,
    assumeCommitted: (image_names: string[], arg: TArg) => TResult
  ) =>
  async (
    arg: TArg,
    { dispatch }: { dispatch: (action: ReturnType<typeof api.util.invalidateTags>) => unknown },
    _extraOptions: unknown,
    baseQuery: ImagesBaseQuery
  ) => {
    const results: TResult[] = [];
    const authContext = captureAuthContext();
    const chunks = chunkImageNames(arg.image_names);
    for (const [index, image_names] of chunks.entries()) {
      const response = await fetchChunk(baseQuery, authContext, { ...request(arg), body: { ...arg, image_names } });
      if (response.error) {
        if (isAuthChangedError(response.error)) {
          // A prior chunk may have committed, but its result belongs to the old session. Do not
          // invalidate the new session or return partial data that its UI can apply.
          return { error: response.error };
        }
        if (isIndeterminateError(response.error)) {
          // A transport-shaped failure proves only that the report was lost: the server may
          // have committed this chunk with nothing coming back to say so. The names still go
          // to failed_images below -- the honest reading, and retrying a name the server
          // already satisfied is safe on every batch route -- but the caches this chunk may
          // have touched must not keep serving the pre-request state until the user notices,
          // so they are invalidated as if the chunk had landed: the refetch shows the truth
          // either way. `assumeCommitted` builds the result this chunk would have returned on
          // success, so the tags come from the same `getTags` the endpoint publishes and the
          // two cannot drift. A lost chunk's affected boards are unknowable, so callers
          // assume none — which reaches the global gallery-list tags but none of the
          // board-keyed ones (`Board`, `BoardImagesTotal`, ..., all keyed by id). Those are
          // appended type-wide instead: every board's sidebar count refetching once beats a
          // count that stays wrong until something unrelated bumps it.
          dispatch(
            api.util.invalidateTags([
              ...getTags(assumeCommitted(image_names, arg)),
              'ImageList',
              'Board',
              'BoardImagesTotal',
              'BoardVideosTotal',
            ])
          );
        }
        if (results.length === 0) {
          // Nothing was applied, so this is an ordinary failed request — report it as one.
          return { error: response.error };
        }
        // Everything from this chunk on is unreached, not merely un-reported.
        const unreached = chunks.slice(index).flat();
        const merged = mergeImageBatchResults(results);
        dispatch(api.util.invalidateTags(getTags(merged)));
        return { data: { ...merged, failed_images: uniq(merged.failed_images.concat(unreached)) } as TResult };
      }
      results.push(response.data as TResult);
    }
    return { data: mergeImageBatchResults(results) };
  };

/**
 * Tag sets for the chunked batch mutations. Extracted from the endpoints so the chunked
 * `queryFn` can invalidate for the chunks that landed before a mid-run failure using the exact
 * tag set the endpoint publishes on success, rather than a hand-rolled approximation.
 */
const getDeleteImagesTags = (result: components['schemas']['DeleteImagesResult']): InvalidateTagsArg => [
  // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
  // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
  // will force those queries to re-fetch, and the requests will of course 404.
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
  'ImageCollectionCounts',
  { type: 'ImageCollection', id: LIST_TAG },
];

const getStarImagesTags = (result: components['schemas']['StarredImagesResult']): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.starred_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
  'ImageCollectionCounts',
  { type: 'ImageCollection', id: 'starred' },
  { type: 'ImageCollection', id: 'unstarred' },
];

const getUnstarImagesTags = (result: components['schemas']['UnstarredImagesResult']): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.unstarred_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
  'ImageCollectionCounts',
  { type: 'ImageCollection', id: 'starred' },
  { type: 'ImageCollection', id: 'unstarred' },
];

const getAddImagesToBoardTags = (result: components['schemas']['AddImagesToBoardResult']): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.added_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
];

const getRemoveImagesFromBoardTags = (
  result: components['schemas']['RemoveImagesFromBoardResult']
): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.removed_images),
  // A name the zero-row classification reports as failed sits on a board this client did not
  // expect — refetching its DTO is what shows where it actually went. Names folded in
  // client-side for unreached chunks ride along; their refetch merely confirms the cache.
  ...getTagsToInvalidateForImageMutation(result.failed_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
];

/**
 * Publishes fetched DTOs into the individual `getImageDTO` caches, which is where every
 * subscribed component reads from — this batch endpoint holds no cache of its own.
 */
const upsertImageDTOs = (dispatch: Dispatch<UnknownAction>, imageDTOs: ImageDTO[]) => {
  if (imageDTOs.length === 0) {
    return;
  }
  const updates: Param0<typeof imagesApi.util.upsertQueryEntries> = [];
  for (const imageDTO of imageDTOs) {
    updates.push({
      endpointName: 'getImageDTO',
      arg: imageDTO.image_name,
      value: imageDTO,
    });
  }
  dispatch(imagesApi.util.upsertQueryEntries(updates));
};

/**
 * Surfaces the partial-failure warning. Used for both kinds of partial failure: the names the
 * server reported it could not apply, and the names a chunked run never reached.
 */
const toastFailedImages = (count: number) => {
  if (count > 0) {
    toast({
      id: 'IMAGES_FAILED_TO_UPDATE',
      title: i18n.t('toast.imagesFailedToUpdate', { count }),
      status: 'warning',
    });
  }
};

/** Reports a whole batch that failed before its first request could commit. */
export const toastFailedImageBatch = (image_names: string[]) => {
  toastFailedImages(new Set(image_names).size);
};

/**
 * The outcome handler shared by all five batch mutations, which report failure identically.
 *
 * Both branches matter and neither is reachable from anywhere else. `handleDeletions` and its
 * siblings swallow every outcome, so without the fulfilled branch a run that only partly landed
 * — server-side per-name failures, or chunks a mid-run failure never reached — says nothing at
 * all. And on the ordinary failure path `buildChunkedImageBatchQueryFn` returns an error only
 * when `results.length === 0`, i.e. the first chunk failed and the server committed nothing, so
 * the whole argument list is genuinely unapplied and reporting all of it is exact rather than an
 * over-count. Nothing else covers that case: these five endpoints have no `matchRejected`
 * listener, unlike the single-image board routes.
 *
 * Three other paths reach this branch with chunks already committed, and the session check is
 * what makes each safe. A *takeover* aborts the run as an error however much landed —
 * deliberately, so the new session cannot consume the old one's aggregate — and the check below
 * is why that does not surface as the previous user's count. An *expiry* mid-run resolves with
 * partial data instead (see `SESSION_ENDED_ERROR`), so it lands on the fulfilled branch, where
 * the same check keeps the count off the login screen while `handleDeletions` still does the
 * state work off the partial payload. (The queryFn's own `invalidateTags` also runs, but under
 * a same-tab expiry `resetApiState` has already emptied the store by then, so it is a no-op —
 * the pruning is the part that matters.) And the queryFn can *throw* on the mid-run path
 * (`getTags(merged)` and `merged.failed_images.concat(...)` both read keys straight off a
 * server payload), which would over-count; that needs a response missing a documented key, so
 * it is left as an over-report rather than a silence.
 *
 * Extracted rather than repeated inline five times so that the wiring is one unit a test can
 * hold — an endpoint that swallows its rejection looks identical to one that reports it, and
 * that difference is invisible to a test of the toast helpers alone.
 */
export const reportImageBatchOutcome = async (
  { image_names }: { image_names: string[] },
  { queryFulfilled }: { queryFulfilled: Promise<{ data: { failed_images: string[] } }> }
) => {
  const authContext = captureAuthContext();
  try {
    const { data: result } = await queryFulfilled;
    if (!isSameAuthContext(authContext)) {
      return;
    }
    toastFailedImages(result.failed_images.length);
  } catch {
    if (!isSameAuthContext(authContext)) {
      return;
    }
    toastFailedImageBatch(image_names);
  }
};

/**
 * The download counterpart. Distinct id and wording: "could not be updated" is wrong for a
 * download, and sharing the id would let one warning overwrite the other, since the toast
 * system updates in place.
 */
const toastFailedDownloads = (count: number) => {
  if (count > 0) {
    toast({
      id: 'IMAGES_FAILED_TO_DOWNLOAD',
      title: i18n.t('toast.imagesFailedToDownload', { count }),
      status: 'warning',
    });
  }
};

/**
 * Runs `/images/download` one conforming chunk at a time.
 *
 * Chunked like the mutating batch routes, but it cannot merge: each request produces its own
 * bulk-download item, so an oversized selection becomes several zips rather than one. That is
 * the cost of keeping the selection downloadable at all — the alternative is a 422 the moment
 * the user hits select-all on a board past the cap.
 *
 * Only the first item name is returned, and only to give `matchFulfilled` a name for the single
 * "preparing" toast. The zips themselves arrive independently: each background task emits its
 * own `bulk_download_complete`, and the socket handler fetches and saves per event, keyed on the
 * item name in the event rather than on this payload.
 *
 * A mid-run failure therefore cannot be reported as a plain rejection. The route answers 202 the
 * moment it has scheduled the background task, so every chunk before the failing one is already
 * producing a zip that will land in the user's downloads. Rejecting drives `matchRejected` and
 * toasts "problem preparing download" while those zips arrive anyway — the opposite of what
 * happened. So, exactly as in `buildChunkedImageBatchQueryFn`, only a run where *nothing* was
 * scheduled surfaces as an error; a partial run resolves and reports the names that never made
 * it into any zip.
 */
export const bulkDownloadQueryFn = async (
  { image_names, board_id }: components['schemas']['Body_download_images_from_list'],
  _api: unknown,
  _extraOptions: unknown,
  baseQuery: ImagesBaseQuery
) => {
  // A board download expands server-side from board_id alone, so there is nothing to split —
  // and the server picks board_id over image_names when both are set (`BulkDownloadService`),
  // which makes a body carrying both a board download too, not a selection to chunk. Splitting
  // it would ask for the same full-board zip once per chunk: 1001 names alongside a board id
  // would schedule two identical full-board jobs. Normalized here to the one request the server
  // is actually going to honour, so the client cannot amplify a download by sending a field the
  // server ignores. No UI caller sends both today; the endpoint accepts it.
  const chunks: (string[] | undefined)[] = board_id ? [undefined] : chunkImageNames(image_names ?? []);
  const authContext = captureAuthContext();
  // Tracked separately from the payload rather than inferred from it. `fetchBaseQuery` resolves
  // an empty response entity as `data: null`, so a 202 whose body did not survive the trip back
  // leaves nothing to return even though the background task WAS scheduled and its zip is
  // already being built. Any nullish test over the payload then sends us down the
  // nothing-happened path and toasts a failure over an arriving download.
  let scheduled = false;
  let first: components['schemas']['ImagesDownloaded'] | undefined;
  for (const [index, chunk] of chunks.entries()) {
    const response = await fetchChunk(baseQuery, authContext, {
      url: buildImagesUrl('download'),
      method: 'POST',
      body: { image_names: chunk, board_id },
    });
    if (response.error) {
      // The mismatch errors from the pre/post-request checks, AND any error response that came
      // back into a session that is no longer the one that asked — which is how this chunk's
      // OWN expired-session 401 arrives: `dynamicBaseQuery` dispatches `sessionExpiredLogout`
      // before returning it, so the token is already gone. The mutating loops deliberately do
      // NOT reclassify that 401 — they need it on the partial path, where the payload feeds
      // `handleDeletions` — but a download has no state work to salvage, and its usual outputs
      // are both wrong for a session that is ending: the failure count is a miscount at the
      // login screen, and returning `first` drives `matchFulfilled` into raising the
      // "preparing" toast with `duration: null` — dismissed only by a socket event this
      // session will never receive.
      if (isSessionMismatchError(response.error) || !isSameAuthContext(authContext)) {
        // One thing IS lost on expiry and deserves saying so: zips already scheduled keep
        // building server-side, but their completion events fire into a socket this session is
        // tearing down, so nothing will ever offer them. Until scheduled downloads are queued
        // per account and replayed after re-authentication (follow-up), the honest move is a
        // plain, finite toast telling the user to re-run — not silence they will read as a
        // download that never came. A takeover (someone else's token in localStorage) stays
        // fully silent: the interruption belongs to whoever started the download, never to the
        // user who takes the tab over. And an expiry with nothing scheduled has lost nothing.
        if (localStorage.getItem('auth_token') === null && scheduled) {
          toast({
            id: 'DOWNLOADS_INTERRUPTED',
            title: i18n.t('toast.downloadsInterrupted'),
            status: 'warning',
          });
        }
        return { data: undefined as unknown as components['schemas']['ImagesDownloaded'] };
      }
      if (!scheduled) {
        // Nothing was scheduled, so this is an ordinary failed request — report it as one and
        // let the `matchRejected` listener raise the failure toast.
        return { error: response.error };
      }
      // Everything from this chunk on is unreached: those names are in no zip. The failing
      // chunk counts too — it was reached, but nothing was scheduled for it. Toasted here
      // rather than folded into the payload because `ImagesDownloaded` has no per-name failure
      // list, and unlike the mutating routes this endpoint has no `onQueryStarted` that would
      // make this a second reporting site for the same toast id.
      toastFailedDownloads(chunks.slice(index).flatMap((names) => names ?? []).length);
      // Cast, like the return below: `first` is nullish only when a 202 came back with no body
      // (see `scheduled`), and ImagesDownloaded has no way to say "scheduled, but the body did
      // not survive the trip". The fulfilled-action listener reads the payload through
      // optionals for exactly that reason.
      return { data: first as components['schemas']['ImagesDownloaded'] };
    }
    scheduled = true;
    first ??= response.data as components['schemas']['ImagesDownloaded'];
  }
  // The final chunk has no next iteration to catch an expiry for it. `fetchChunk`'s own
  // post-response check deliberately passes a mere expiry through -- the mutating loops need
  // that -- so a token cleared during the last chunk's await (a 401 on any concurrent request
  // does it) would otherwise sail into the return below, and `matchFulfilled` would raise the
  // `duration: null` "preparing" toast into a session whose socket will never deliver the
  // dismissal -- or the zips. Same triage as the in-loop expiry arm: say what was lost, and
  // hand `matchFulfilled` nothing to toast on. Synchronous from here to the return, so there
  // is no later window this check misses.
  if (!isSameAuthContext(authContext)) {
    if (localStorage.getItem('auth_token') === null && scheduled) {
      toast({
        id: 'DOWNLOADS_INTERRUPTED',
        title: i18n.t('toast.downloadsInterrupted'),
        status: 'warning',
      });
    }
    return { data: undefined as unknown as components['schemas']['ImagesDownloaded'] };
  }
  return { data: first as components['schemas']['ImagesDownloaded'] };
};

/**
 * Fetches DTOs for a list of image names, one conforming chunk at a time.
 *
 * Chunked despite being a read: `useRangeBasedImageFetching` unions every virtuoso range seen
 * inside its throttle window, and a dense grid with a 4096px overscan can carry that past the
 * cap. The failure was silent — the hook swallows the rejection and nothing listens for it, so
 * the affected thumbnails simply never loaded.
 *
 * Each chunk is published into the individual `getImageDTO` caches as it arrives, rather than
 * all of them at the end. This mutation rejects on any chunk failure and its one caller never
 * looks at the rejection, so a transient failure late in a wide range would otherwise throw
 * away every thumbnail the earlier chunks had already fetched — and they would not be
 * re-requested either, since the hook asks only for names it cannot find in the cache and the
 * next request is driven by scrolling, not by the failure.
 */
export const imageDTOsByNamesQueryFn = async (
  { image_names }: components['schemas']['Body_get_images_by_names'],
  // Typed as the plain store dispatch rather than against `imagesApi.util.upsertQueryEntries`:
  // naming imagesApi in a signature the endpoint definitions themselves depend on makes its
  // inferred type circular, and every consumer of the api silently degrades to `any`.
  { dispatch }: { dispatch: Dispatch<UnknownAction> },
  _extraOptions: unknown,
  baseQuery: ImagesBaseQuery
) => {
  const authContext = captureAuthContext();
  const imageDTOs: ImageDTO[] = [];
  for (const chunk of chunkImageNames(image_names)) {
    const response = await fetchChunk(baseQuery, authContext, {
      url: buildImagesUrl('images_by_names'),
      method: 'POST',
      body: { image_names: chunk },
    });
    if (response.error) {
      return { error: response.error };
    }
    const chunkDTOs = response.data as ImageDTO[];
    // Re-checked here as well as inside `fetchChunk`, and not folded into it: `fetchChunk` is
    // async, so resuming from it is a microtask hop, and a logout that lands in that hop passes
    // its check and still clears the cache before this line. Only a check with no await between
    // it and the write closes that. Publishing then would seed one user's cache with another's
    // images, into a store the logout listener has just reset.
    if (isSameAuthContext(authContext)) {
      upsertImageDTOs(dispatch, chunkDTOs);
    }
    imageDTOs.push(...chunkDTOs);
  }
  return { data: imageDTOs };
};

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    listImages: build.query<ListImagesResponse, ListImagesArgs>({
      query: (queryArgs) => ({
        // Use the helper to create the URL.
        url: getListImagesUrl(queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => {
        return [
          // Make the tags the same as the cache key
          { type: 'ImageList', id: stableHash(queryArgs) },
          { type: 'Board', id: queryArgs.board_id ?? 'none' },
          'FetchOnReconnect',
        ];
      },
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        // Populate the getImageDTO cache with these images. This makes image selection smoother, because it doesn't
        // need to re-fetch image data when the user selects an image. The getImageDTO cache keeps data for the default
        // of 60s, so this data won't stick around too long.
        const res = await queryFulfilled;
        const imageDTOs = res.data.items;
        const updates: Param0<typeof imagesApi.util.upsertQueryEntries> = [];
        for (const imageDTO of imageDTOs) {
          updates.push({
            endpointName: 'getImageDTO',
            arg: imageDTO.image_name,
            value: imageDTO,
          });
        }
        dispatch(imagesApi.util.upsertQueryEntries(updates));
      },
    }),
    getIntermediatesCount: build.query<number, void>({
      query: () => ({ url: buildImagesUrl('intermediates') }),
      providesTags: ['IntermediatesCount', 'FetchOnReconnect'],
    }),
    clearIntermediates: build.mutation<number, void>({
      query: () => ({ url: buildImagesUrl('intermediates'), method: 'DELETE' }),
      invalidatesTags: [
        'IntermediatesCount',
        'InvocationCacheStatus',
        'ImageCollectionCounts',
        { type: 'ImageCollection', id: LIST_TAG },
      ],
    }),
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}`) }),
      providesTags: (result, error, image_name) => [{ type: 'Image', id: image_name }],
    }),
    getImageMetadata: build.query<JsonObject | undefined, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}/metadata`) }),
      providesTags: (result, error, image_name) => [{ type: 'ImageMetadata', id: image_name }],
    }),
    getImageWorkflow: build.query<GraphAndWorkflowResponse, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}/workflow`) }),
      providesTags: (result, error, image_name) => [{ type: 'ImageWorkflow', id: image_name }],
    }),
    deleteImage: build.mutation<
      paths['/api/v1/images/i/{image_name}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/i/{image_name}']['delete']['parameters']['path']
    >({
      query: ({ image_name }) => ({
        url: buildImagesUrl(`i/${image_name}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
        // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
        // will force those queries to re-fetch, and the requests will of course 404.
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
        ];
      },
    }),
    deleteImages: build.mutation<
      paths['/api/v1/images/delete']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/delete']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildImagesUrl('delete'), method: 'POST' }),
        getDeleteImagesTags,
        (image_names) => ({ deleted_images: image_names, failed_images: [], affected_boards: [] })
      ),
      onQueryStarted: reportImageBatchOutcome,
      invalidatesTags: (result) => (result ? getDeleteImagesTags(result) : []),
    }),
    deleteUncategorizedImages: build.mutation<
      paths['/api/v1/images/uncategorized']['delete']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({ url: buildImagesUrl('uncategorized'), method: 'DELETE' }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
        // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
        // will force those queries to re-fetch, and the requests will of course 404.
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
        ];
      },
    }),
    /**
     * Change an image's `is_intermediate` property.
     */
    changeImageIsIntermediate: build.mutation<
      paths['/api/v1/images/i/{image_name}']['patch']['responses']['200']['content']['application/json'],
      { image_name: string; is_intermediate: boolean }
    >({
      query: ({ image_name, is_intermediate }) => ({
        url: buildImagesUrl(`i/${image_name}`),
        method: 'PATCH',
        body: { is_intermediate },
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation([result.image_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([result.board_id ?? 'none']),
        ];
      },
    }),
    /**
     * Star a list of images.
     */
    starImages: build.mutation<
      paths['/api/v1/images/star']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/star']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildImagesUrl('star'), method: 'POST' }),
        getStarImagesTags,
        (image_names) => ({ starred_images: image_names, failed_images: [], affected_boards: [] })
      ),
      onQueryStarted: reportImageBatchOutcome,
      invalidatesTags: (result) => (result ? getStarImagesTags(result) : []),
    }),
    /**
     * Unstar a list of images.
     */
    unstarImages: build.mutation<
      paths['/api/v1/images/unstar']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/unstar']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildImagesUrl('unstar'), method: 'POST' }),
        getUnstarImagesTags,
        (image_names) => ({ unstarred_images: image_names, failed_images: [], affected_boards: [] })
      ),
      onQueryStarted: reportImageBatchOutcome,
      invalidatesTags: (result) => (result ? getUnstarImagesTags(result) : []),
    }),
    uploadImage: build.mutation<
      paths['/api/v1/images/upload']['post']['responses']['201']['content']['application/json'],
      UploadImageArg
    >({
      query: ({ file, image_category, is_intermediate, session_id, board_id, crop_visible, metadata, resize_to }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }
        if (resize_to) {
          formData.append('resize_to', JSON.stringify(resize_to));
        }
        return {
          url: buildImagesUrl('upload'),
          method: 'POST',
          body: formData,
          params: {
            image_category,
            is_intermediate,
            session_id,
            board_id: board_id === 'none' ? undefined : board_id,
            crop_visible,
          },
        };
      },
      invalidatesTags: (result) => {
        if (!result || result.is_intermediate) {
          // Don't add it to anything
          return [];
        }
        const boardId = result.board_id ?? 'none';

        return [
          ...getTagsToInvalidateForImageMutation([result.image_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([boardId]),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
          'ImageNameList',
        ];
      },
    }),
    createImageUploadEntry: build.mutation<ImageUploadEntryResponse, ImageUploadEntryRequest>({
      query: ({ width, height, board_id }) => ({
        url: buildImagesUrl(),
        method: 'POST',
        body: { width, height, board_id },
      }),
    }),
    deleteBoard: build.mutation<
      paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/boards/{board_id}']['delete']['parameters']['path']
    >({
      query: ({ board_id }) => ({ url: buildBoardsUrl(board_id), method: 'DELETE' }),
      invalidatesTags: (result) => [
        { type: 'Board', id: LIST_TAG },
        // Both images and videos on the board cascade to the 'No Board' bucket on the
        // backend side; invalidate the 'none' caches for both kinds so the polymorphic
        // gallery surfaces them. The Gallery* tags refresh the unified gallery list view.
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: IMAGE_CATEGORIES,
          }),
        },
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: ASSETS_CATEGORIES,
          }),
        },
        { type: 'VideoList', id: LIST_TAG },
        'VideoNameList',
        'GalleryItemList',
        'GalleryItemNameList',
        // The orphaned media keep cached DTOs whose board_id still points at the deleted
        // board; refetch them so drag/drop and context menus see the new 'none' board.
        ...getTagsToInvalidateForImageMutation(result?.deleted_board_images ?? []),
        ...getTagsToInvalidateForVideoMutation(result?.deleted_board_videos ?? []),
      ],
    }),

    deleteBoardAndImages: build.mutation<
      paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/boards/{board_id}']['delete']['parameters']['path']
    >({
      query: ({ board_id }) => ({
        url: buildBoardsUrl(board_id),
        method: 'DELETE',
        params: { include_images: true },
      }),
      // The backend now also cascade-deletes videos on the board, so the unified gallery
      // and the video list both need invalidation in addition to the board tag.
      invalidatesTags: (result) => [
        { type: 'Board', id: LIST_TAG },
        'VirtualBoards',
        { type: 'VideoList', id: LIST_TAG },
        'VideoNameList',
        'GalleryItemList',
        'GalleryItemNameList',
        ...getTagsToInvalidateForBoardAffectingMutation(['none']),
        // Deleted media must drop out of their per-item caches (so DTO queries become
        // 404s instead of serving stale entries, e.g. to node inputs still referencing
        // them). Only server-confirmed deletions are listed in the result.
        ...getTagsToInvalidateForImageMutation(result?.deleted_images ?? []),
        ...getTagsToInvalidateForVideoMutation(result?.deleted_videos ?? []),
        ...getTagsToInvalidateForImageMutation(result?.failed_images ?? []),
        ...getTagsToInvalidateForVideoMutation(result?.failed_videos ?? []),
      ],
    }),
    addImageToBoard: build.mutation<
      paths['/api/v1/board_images/']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => {
        return {
          url: buildBoardImagesUrl(),
          method: 'POST',
          body,
        };
      },
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.added_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    removeImageFromBoard: build.mutation<
      paths['/api/v1/board_images/']['delete']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/']['delete']['requestBody']['content']['application/json']
    >({
      query: (body) => {
        return {
          url: buildBoardImagesUrl(),
          method: 'DELETE',
          body,
        };
      },
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.removed_images),
          // A name the zero-row classification reported as failed sits on a board this client
          // did not expect; refetching its DTO is what shows where it actually went.
          ...getTagsToInvalidateForImageMutation(result.failed_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    addImagesToBoard: build.mutation<
      paths['/api/v1/board_images/batch']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/batch']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildBoardImagesUrl('batch'), method: 'POST' }),
        getAddImagesToBoardTags,
        (image_names, arg) => ({ added_images: image_names, failed_images: [], affected_boards: [arg.board_id] })
      ),
      onQueryStarted: reportImageBatchOutcome,
      invalidatesTags: (result) => (result ? getAddImagesToBoardTags(result) : []),
    }),
    removeImagesFromBoard: build.mutation<
      paths['/api/v1/board_images/batch/delete']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/batch/delete']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildBoardImagesUrl('batch/delete'), method: 'POST' }),
        getRemoveImagesFromBoardTags,
        (image_names) => ({ removed_images: image_names, failed_images: [], affected_boards: [] })
      ),
      onQueryStarted: reportImageBatchOutcome,
      invalidatesTags: (result) => (result ? getRemoveImagesFromBoardTags(result) : []),
    }),
    bulkDownloadImages: build.mutation<
      components['schemas']['ImagesDownloaded'],
      components['schemas']['Body_download_images_from_list']
    >({
      queryFn: bulkDownloadQueryFn,
    }),
    /**
     * Get ordered list of image names for selection operations
     */
    getImageNames: build.query<GetImageNamesResult, GetImageNamesArgs>({
      query: (queryArgs) => ({
        url: buildImagesUrl('names', queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        'ImageNameList',
        'FetchOnReconnect',
        { type: 'ImageNameList', id: stableHash(queryArgs) },
      ],
    }),
    /**
     * Get image DTOs for the specified image names. Maintains order of input names.
     */
    getImageDTOsByNames: build.mutation<
      paths['/api/v1/images/images_by_names']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/images_by_names']['post']['requestBody']['content']['application/json']
    >({
      queryFn: imageDTOsByNamesQueryFn,
      // No cache tags: the DTOs are upserted into the individual getImageDTO caches by the
      // queryFn, chunk by chunk, which is also all an onQueryStarted handler would have done.
    }),
  }),
});

export const {
  useGetIntermediatesCountQuery,
  useListImagesQuery,
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
  useGetImageWorkflowQuery,
  useLazyGetImageWorkflowQuery,
  useUploadImageMutation,
  useClearIntermediatesMutation,
  useAddImagesToBoardMutation,
  useRemoveImagesFromBoardMutation,
  useDeleteBoardAndImagesMutation,
  useDeleteUncategorizedImagesMutation,
  useDeleteBoardMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
  useBulkDownloadImagesMutation,
  useGetImageDTOsByNamesMutation,
} = imagesApi;

/**
 * Imperative RTKQ helper to fetch an ImageDTO.
 * @param image_name The name of the image to fetch
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @returns The ImageDTO if found, otherwise null
 */
export const getImageDTOSafe = async (
  image_name: string,
  options?: Parameters<typeof imagesApi.endpoints.getImageDTO.initiate>[1]
): Promise<ImageDTO | null> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(imagesApi.endpoints.getImageDTO.initiate(image_name, _options));
  try {
    return await req.unwrap();
  } catch {
    return null;
  }
};

/**
 * Imperative RTKQ helper to fetch an ImageDTO.
 * @param image_name The name of the image to fetch
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @raises Error if the image is not found or there is an error fetching the image
 */
export const getImageDTO = (
  image_name: string,
  options?: Parameters<typeof imagesApi.endpoints.getImageDTO.initiate>[1]
): Promise<ImageDTO> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(imagesApi.endpoints.getImageDTO.initiate(image_name, _options));
  return req.unwrap();
};

export const uploadImage = (arg: UploadImageArg): Promise<ImageDTO> => {
  const { dispatch } = getStore();
  const req = dispatch(imagesApi.endpoints.uploadImage.initiate(arg, { track: false }));
  return req.unwrap();
};

export const copyImage = async (imageName: string, uploadImageArg: Omit<UploadImageArg, 'file'>): Promise<ImageDTO> => {
  const originalImageDTO = await getImageDTO(imageName);
  const file = await imageDTOToFile(originalImageDTO);
  const imageDTO = await uploadImage({ file, ...uploadImageArg });
  return imageDTO;
};

export const uploadImages = async (args: UploadImageArg[]): Promise<ImageDTO[]> => {
  const { dispatch } = getStore();
  const results = await Promise.allSettled(
    args.map((arg) => {
      const req = dispatch(imagesApi.endpoints.uploadImage.initiate(arg, { track: false }));
      return req.unwrap();
    })
  );
  return results.filter((r): r is PromiseFulfilledResult<ImageDTO> => r.status === 'fulfilled').map((r) => r.value);
};

/**
 * Convert an ImageDTO to a File by downloading the image from the server.
 * @param imageDTO The image to download and convert to a File
 */
export const imageDTOToFile = async (imageDTO: ImageDTO): Promise<File> => {
  const init: RequestInit = {};
  const res = await fetch(imageDTO.image_url, init);
  const blob = await res.blob();
  // Create a new file with the same name, which we will upload
  const file = new File([blob], `copy_of_${imageDTO.image_name}`, { type: 'image/png' });
  return file;
};

export const useImageDTO = (imageName: string | null | undefined) => {
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);
  return imageDTO ?? null;
};

export const useImageDTOFromCroppableImage = (croppableImage: CroppableImageWithDims | null) => {
  const { currentData: imageDTO } = useGetImageDTOQuery(
    croppableImage?.crop?.image.image_name ?? croppableImage?.original.image.image_name ?? skipToken
  );
  return imageDTO ?? null;
};
