import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { toast } from 'features/toast/toast';
import i18n from 'i18next';
import {
  buildChunkedImageBatchQueryFn,
  bulkDownloadQueryFn,
  chunkImageNames,
  imageDTOsByNamesQueryFn,
  imagesApi,
  mergeImageBatchResults,
  reportImageBatchOutcome,
  toastFailedImageBatch,
} from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '..';

vi.mock('features/toast/toast', () => ({ toast: vi.fn() }));
vi.mock('i18next', () => ({ default: { t: vi.fn((key: string) => key) } }));

/** Mirrors MAX_IMAGE_BATCH_SIZE in invokeai/app/api/routers/images.py. */
const CHUNK_SIZE = 1000;

const names = (count: number) => Array.from({ length: count }, (_, i) => `image-${i}.png`);

/**
 * The chunk loops read the live session out of localStorage before every request, so the tests
 * need a real one to write to. Not available in the default (node) test environment.
 */
beforeAll(() => {
  const values = new Map<string, string>();
  vi.stubGlobal('localStorage', {
    clear: () => values.clear(),
    getItem: (key: string) => values.get(key) ?? null,
    key: (index: number) => [...values.keys()][index] ?? null,
    get length() {
      return values.size;
    },
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  });
});

beforeEach(() => {
  localStorage.clear();
});

/** A token shaped like the real JWT: only the `user_id` claim is read back out of it. */
const tokenFor = (userId: string, issuedAt = 1) =>
  `header.${btoa(JSON.stringify({ user_id: userId, iat: issuedAt }))}.signature`;

const login = (userId: string, issuedAt = 1) => {
  localStorage.setItem('auth_token', tokenFor(userId, issuedAt));
};

/** What a logout + login as someone else leaves behind: a new token AND a bumped generation. */
const switchUser = (userId: string) => {
  localStorage.setItem('auth_generation', String(Number(localStorage.getItem('auth_generation') ?? 0) + 1));
  login(userId);
};

describe('chunkImageNames', () => {
  it('leaves a conforming list as a single request', () => {
    expect(chunkImageNames(names(3))).toEqual([names(3)]);
    expect(chunkImageNames(names(CHUNK_SIZE))).toHaveLength(1);
  });

  it('still issues one request for an empty list', () => {
    // The routes answer an empty body with a well-formed empty result. Merging zero chunks
    // would produce `{}`, and callers read fields like `failed_images` off the result.
    expect(chunkImageNames([])).toEqual([[]]);
  });

  it('splits an oversized list into conforming chunks that cover it exactly', () => {
    const all = names(2500);
    const chunks = chunkImageNames(all);
    expect(chunks.map((c) => c.length)).toEqual([1000, 1000, 500]);
    expect(chunks.flat()).toEqual(all);
  });
});

describe('mergeImageBatchResults', () => {
  it('unions each key and dedupes, so the caller sees one single-request-shaped result', () => {
    expect(
      mergeImageBatchResults([
        { deleted_images: ['a.png'], failed_images: [], affected_boards: ['board-1', 'none'] },
        { deleted_images: ['b.png'], failed_images: ['c.png'], affected_boards: ['board-1'] },
      ])
    ).toEqual({
      deleted_images: ['a.png', 'b.png'],
      failed_images: ['c.png'],
      affected_boards: ['board-1', 'none'],
    });
  });
});

describe('toastFailedImageBatch', () => {
  beforeEach(() => {
    vi.mocked(toast).mockClear();
    vi.mocked(i18n.t).mockClear();
  });

  it('reports every unique name when the first request fails', () => {
    toastFailedImageBatch(['a.png', 'a.png', 'b.png']);

    expect(i18n.t).toHaveBeenCalledWith('toast.imagesFailedToUpdate', { count: 2 });
    expect(toast).toHaveBeenCalledWith({
      id: 'IMAGES_FAILED_TO_UPDATE',
      title: 'toast.imagesFailedToUpdate',
      status: 'warning',
    });
  });
});

describe('reportImageBatchOutcome', () => {
  beforeEach(() => {
    vi.mocked(toast).mockClear();
    vi.mocked(i18n.t).mockClear();
  });

  it('reports the names the server could not apply when the request resolves', async () => {
    await reportImageBatchOutcome(
      { image_names: names(3) },
      { queryFulfilled: Promise.resolve({ data: { failed_images: ['image-1.png'] } }) }
    );

    expect(i18n.t).toHaveBeenCalledWith('toast.imagesFailedToUpdate', { count: 1 });
  });

  it('says nothing when every name was applied', async () => {
    await reportImageBatchOutcome(
      { image_names: names(3) },
      { queryFulfilled: Promise.resolve({ data: { failed_images: [] } }) }
    );

    expect(toast).not.toHaveBeenCalled();
  });

  it('reports the whole argument list when the request rejects', async () => {
    // A rejection out of the chunked queryFn is raised only when nothing was committed, so
    // every name really is unapplied. Swallowing it leaves a delete or a move that landed
    // nothing saying nothing at all -- these endpoints have no matchRejected listener.
    await reportImageBatchOutcome({ image_names: names(3) }, { queryFulfilled: Promise.reject(new Error('boom')) });

    expect(i18n.t).toHaveBeenCalledWith('toast.imagesFailedToUpdate', { count: 3 });
    expect(toast).toHaveBeenCalledWith({
      id: 'IMAGES_FAILED_TO_UPDATE',
      title: 'toast.imagesFailedToUpdate',
      status: 'warning',
    });
  });

  it('does not report an outcome from a session that ended while it was pending', async () => {
    login('user-a');
    let resolveQuery: (value: { data: { failed_images: string[] } }) => void = () => {};
    const queryFulfilled = new Promise<{ data: { failed_images: string[] } }>((resolve) => {
      resolveQuery = resolve;
    });

    const outcome = reportImageBatchOutcome({ image_names: names(3) }, { queryFulfilled });
    switchUser('user-b');
    resolveQuery({ data: { failed_images: ['image-1.png'] } });
    await outcome;

    expect(toast).not.toHaveBeenCalled();
  });

  it('does not report a rejection from a session that ended while it was pending', async () => {
    // The branch the session change actually takes: an auth-changed abort comes back as an
    // error, so `queryFulfilled` rejects. Guarding only the fulfilled branch would leave the
    // previous user's whole selection toasted at whoever holds the tab next.
    login('user-a');
    let rejectQuery: (reason: unknown) => void = () => {};
    const queryFulfilled = new Promise<{ data: { failed_images: string[] } }>((_resolve, reject) => {
      rejectQuery = reject;
    });

    const outcome = reportImageBatchOutcome({ image_names: names(3) }, { queryFulfilled });
    switchUser('user-b');
    rejectQuery(new Error('aborted'));
    await outcome;

    expect(toast).not.toHaveBeenCalled();
  });

  it('is wired into every chunked batch mutation', () => {
    // RTK exposes no way to reach an endpoint's `onQueryStarted` at runtime -- the built
    // endpoint object carries only initiate/select/match*/hooks -- so the wiring is guarded at
    // the source level, as elsewhere in this repo. Counted against the chunked endpoints rather
    // than a fixed number, so a sixth one that forgets to report its failures fails this. The
    // call sites are matched wherever they are, not only inline after `queryFn:`, since hoisting
    // one to a const is otherwise enough to slip an unreporting endpoint past this.
    const source = readFileSync(fileURLToPath(new URL('./images.ts', import.meta.url)), 'utf8');
    const chunked = source.match(/buildChunkedImageBatchQueryFn\(/g) ?? [];
    const wired = source.match(/onQueryStarted: reportImageBatchOutcome,/g) ?? [];

    expect(chunked.length).toBeGreaterThan(0);
    expect(wired).toHaveLength(chunked.length);
  });
});

describe('buildChunkedImageBatchQueryFn', () => {
  type Arg = { image_names: string[]; board_id?: string };
  type Result = { added_images: string[]; failed_images: string[]; affected_boards: string[] };
  type Request = { url: string; method: string; body: Arg };
  type Response = { data: Result } | { error: { status: number; data: string } };

  const getTags = () => ['ImageCollectionCounts' as const];

  beforeEach(() => {
    vi.mocked(toast).mockClear();
  });

  const run = (baseQuery: (args: Request) => Promise<Response>, arg: Arg) => {
    const dispatch = vi.fn();
    const queryFn = buildChunkedImageBatchQueryFn<Result, Arg>(
      () => ({ url: '/api/v1/board_images/batch', method: 'POST' }),
      getTags
    );
    /* eslint-disable @typescript-eslint/no-explicit-any */
    return { dispatch, result: queryFn(arg, { dispatch } as any, undefined, baseQuery as any) };
    /* eslint-enable @typescript-eslint/no-explicit-any */
  };

  it('sends one request per chunk, carrying the non-name body fields on each', async () => {
    const baseQuery = vi.fn(
      (_args: Request): Promise<Response> =>
        Promise.resolve({ data: { added_images: [], failed_images: [], affected_boards: [] } })
    );

    const { result } = run(baseQuery, { image_names: names(2500), board_id: 'board-1' });
    await result;

    expect(baseQuery).toHaveBeenCalledTimes(3);
    // board_id must ride along with every chunk, not just the first.
    expect(baseQuery.mock.calls.map(([args]) => args.body.board_id)).toEqual(['board-1', 'board-1', 'board-1']);
    expect(baseQuery.mock.calls.map(([args]) => args.body.image_names.length)).toEqual([1000, 1000, 500]);
  });

  it('merges the per-chunk results into one aggregate', async () => {
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      return Promise.resolve({
        data: { added_images: [`chunk-${call}.png`], failed_images: [], affected_boards: ['board-1'] },
      });
    });

    const { result } = run(baseQuery, { image_names: names(1500) });

    expect(await result).toEqual({
      data: { added_images: ['chunk-1.png', 'chunk-2.png'], failed_images: [], affected_boards: ['board-1'] },
    });
  });

  it('reports a mid-run failure as a partial success, keeping what the server already applied', async () => {
    // A bare error would discard the first chunk's payload. It is not only the RTK cache at
    // stake: handleDeletions drives the gallery selection and strips deleted images out of
    // nodes, canvas layers and reference images off `deleted_images`, and none of that runs on
    // a rejection — so 1000 images would be gone from the DB and still referenced by the canvas.
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      if (call === 3) {
        return Promise.resolve({ error: { status: 500, data: 'boom' } });
      }
      return Promise.resolve({
        data: { added_images: [`chunk-${call}.png`], failed_images: [], affected_boards: ['board-1'] },
      });
    });

    const { dispatch, result } = run(baseQuery, { image_names: names(2500) });

    // The failing chunk's 500 names are unreached, not merely un-reported, so they are folded
    // into failed_images -- one place, so the endpoint's single toast reports one true total.
    expect(await result).toEqual({
      data: {
        added_images: ['chunk-1.png', 'chunk-2.png'],
        failed_images: names(2500).slice(2000),
        affected_boards: ['board-1'],
      },
    });
    expect(baseQuery).toHaveBeenCalledTimes(3); // stopped, did not keep firing chunks
    expect(dispatch).toHaveBeenCalledWith(api.util.invalidateTags(getTags()));
    // Toasting from here as well would fire twice on one toast id, and the toast system
    // updates in place -- the second count would replace the first rather than adding to it.
    expect(toast).not.toHaveBeenCalled();
  });

  it('reports an error when the first chunk fails, since nothing was applied', async () => {
    const baseQuery = vi.fn(
      (_args: Request): Promise<Response> => Promise.resolve({ error: { status: 403, data: 'nope' } })
    );

    const { dispatch, result } = run(baseQuery, { image_names: names(1500) });

    expect(await result).toEqual({ error: { status: 403, data: 'nope' } });
    expect(dispatch).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
  });

  it('stops when the session changes mid-run, instead of applying the rest as the new user', async () => {
    // Every request picks up whatever token localStorage holds when it is sent, so a loop that
    // outlives its own session finishes as whoever logged in next -- on a public board those
    // writes land, committing half of one user's action under another's name. resetApiState on
    // the logout action does not help: it clears the cache, not a queryFn that is running.
    login('user-a');
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      if (call === 2) {
        switchUser('user-b');
      }
      return Promise.resolve({
        data: { added_images: [`chunk-${call}.png`], failed_images: [], affected_boards: ['board-1'] },
      });
    });

    const { dispatch, result } = run(baseQuery, { image_names: names(2500) });

    // The second response is stale as soon as the session changes, so neither it nor a partial
    // aggregate may reach the new session. The server-side write cannot be rolled back here, but
    // the next session must refetch its own state rather than consume the old result.
    expect(await result).toEqual({
      error: { status: 'CUSTOM_ERROR', error: expect.stringContaining('Aborted') },
    });
    expect(baseQuery).toHaveBeenCalledTimes(2);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('discards a response that arrived after the session changed', async () => {
    // A single-chunk run, so there is no following chunk whose pre-request check could catch
    // this: the response itself came back into a session that is no longer the one that asked
    // for it. Without the check after the response, its payload is returned and applied.
    login('user-a');
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      switchUser('user-b');
      return Promise.resolve({ data: { added_images: ['a.png'], failed_images: [], affected_boards: ['board-1'] } });
    });

    const { dispatch, result } = run(baseQuery, { image_names: names(5) });

    expect(baseQuery).toHaveBeenCalledTimes(1);
    expect(await result).toEqual({ error: { status: 'CUSTOM_ERROR', error: expect.stringContaining('Aborted') } });
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('reports an expired session as the failure it is, not as an abort', async () => {
    // `dynamicBaseQuery` dispatches `sessionExpiredLogout` on a 401 before it returns, and that
    // clears the token synchronously — so by the time the post-response check runs the session
    // has "changed" for every 401 there is. Rewriting those as aborts would make the ordinary
    // expired-session case fatal to the whole run, skipping the partial-success path that
    // reports what the server already committed.
    login('user-a');
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      if (call === 2) {
        localStorage.removeItem('auth_token');
        return Promise.resolve({ error: { status: 401, data: 'expired' } });
      }
      return Promise.resolve({
        data: { added_images: [`chunk-${call}.png`], failed_images: [], affected_boards: ['board-1'] },
      });
    });

    const { dispatch, result } = run(baseQuery, { image_names: names(2500) });

    // Chunk 1 committed, so this is a partial success and its unreached names are reported --
    // the same shape a mid-run 500 produces, which is what lets `handleDeletions` prune.
    expect(await result).toEqual({
      data: {
        added_images: ['chunk-1.png'],
        failed_images: names(2500).slice(1000),
        affected_boards: ['board-1'],
      },
    });
    expect(dispatch).toHaveBeenCalledWith(api.util.invalidateTags(getTags()));
  });

  it('keeps running when a login request elsewhere bumps the auth generation', async () => {
    // `beginAuthTransition` bumps the shared counter when a login or logout request is *sent*,
    // before anything has changed and whether or not it succeeds — a second tab opening the
    // login page must not abandon this tab's batch. The session is judged by who the next
    // request would go out as, which is unchanged here.
    login('user-a');
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      localStorage.setItem('auth_generation', '7');
      return Promise.resolve({ data: { added_images: [], failed_images: [], affected_boards: [] } });
    });

    const { result } = run(baseQuery, { image_names: names(2500) });
    await result;

    expect(baseQuery).toHaveBeenCalledTimes(3);
  });

  it('keeps running across a sliding-window token refresh, which is the same session', async () => {
    // The middleware mints a fresh token on mutating requests. Comparing tokens byte-for-byte
    // would abandon every batch long enough to be refreshed -- the exact operations chunking
    // exists for.
    login('user-a', 1);
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      login('user-a', 2);
      return Promise.resolve({ data: { added_images: [], failed_images: [], affected_boards: [] } });
    });

    const { result } = run(baseQuery, { image_names: names(2500) });
    await result;

    expect(baseQuery).toHaveBeenCalledTimes(3);
  });

  it('stops when the session simply expires mid-run, which bumps no generation', async () => {
    // sessionExpiredLogout drops the token with no request of its own -- a 401 anywhere, or a
    // token that fails validation on load -- so the generation counter never moves. Without the
    // token half of the check, the loop would run on with no credentials at all.
    login('user-a');
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      localStorage.removeItem('auth_token');
      return Promise.resolve({ data: { added_images: [], failed_images: [], affected_boards: [] } });
    });

    const { result } = run(baseQuery, { image_names: names(2500) });
    await result;

    expect(baseQuery).toHaveBeenCalledTimes(1);
  });
});

describe('bulkDownloadQueryFn', () => {
  type Body = { image_names: string[]; board_id?: string };
  type Request = { url: string; method: string; body: Body };
  type Response = { data: { bulk_download_item_name: string } } | { error: { status: number; data: string } };

  const run = (baseQuery: (args: Request) => Promise<Response>, body: Body) =>
    /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
    bulkDownloadQueryFn(body, undefined, undefined, baseQuery as any);

  beforeEach(() => {
    vi.mocked(toast).mockClear();
    vi.mocked(i18n.t).mockClear();
  });

  it('issues one request per chunk and returns the first item name', async () => {
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      return Promise.resolve({ data: { bulk_download_item_name: `item-${call}.zip` } });
    });

    const result = await run(baseQuery, { image_names: names(2500) });

    expect(baseQuery).toHaveBeenCalledTimes(3);
    // Only the first name is returned; each background task announces its own zip over the
    // socket, so the payload is just a handle for the single "preparing" toast.
    expect(result).toEqual({ data: { bulk_download_item_name: 'item-1.zip' } });
    expect(toast).not.toHaveBeenCalled();
  });

  it('sends one request for a board download, which the server expands itself', async () => {
    const baseQuery = vi.fn(
      (_args: Request): Promise<Response> => Promise.resolve({ data: { bulk_download_item_name: 'board.zip' } })
    );

    await run(baseQuery, { image_names: [], board_id: 'board-1' });

    expect(baseQuery).toHaveBeenCalledTimes(1);
    expect(baseQuery.mock.calls[0]?.[0].body.board_id).toBe('board-1');
  });

  it('resolves a mid-run failure instead of rejecting, since the earlier zips still arrive', async () => {
    // The route answers 202 as soon as it has scheduled the background task, so chunk 1's zip
    // is already being built. Rejecting would drive `matchRejected` -> "problem preparing
    // download" while that zip lands in the user's downloads anyway.
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      if (call === 2) {
        return Promise.resolve({ error: { status: 403, data: 'nope' } });
      }
      return Promise.resolve({ data: { bulk_download_item_name: `item-${call}.zip` } });
    });

    const result = await run(baseQuery, { image_names: names(2500) });

    expect(result).toEqual({ data: { bulk_download_item_name: 'item-1.zip' } });
    expect(baseQuery).toHaveBeenCalledTimes(2); // stopped, did not keep firing chunks
    // The 1500 names from the failing chunk on are in no zip at all -- reported here, because
    // ImagesDownloaded carries no per-name failure list to fold them into.
    expect(toast).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast).mock.calls[0]?.[0]).toMatchObject({
      id: 'IMAGES_FAILED_TO_DOWNLOAD',
      status: 'warning',
    });
    // Asserted on the interpolation rather than the title, since i18n is mocked to echo the
    // key. The count is the whole point of the toast: the failing chunk's 1000 names plus the
    // 500 never sent. Off by one chunk in either direction and the user is told the wrong
    // number of images are missing from their download.
    expect(i18n.t).toHaveBeenCalledWith('toast.imagesFailedToDownload', { count: 1500 });
  });

  it('still schedules-and-reports when the 202 body does not survive the trip back', async () => {
    // `fetchBaseQuery` resolves an empty entity as `data: null`, so a proxy that strips the
    // body off the 202 leaves nothing to return -- but the background task was scheduled all
    // the same and its zip will arrive. Any nullish test over the payload (`!first`) treats
    // that as nothing-happened and toasts an error over an arriving download.
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      if (call === 2) {
        return Promise.resolve({ error: { status: 500, data: 'boom' } });
      }
      /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
      return Promise.resolve({ data: null as any });
    });

    const result = await run(baseQuery, { image_names: names(2500) });

    expect(result).toEqual({ data: null });
    expect(i18n.t).toHaveBeenCalledWith('toast.imagesFailedToDownload', { count: 1500 });
  });

  it('sends one request when a board id rides along with an oversized name list', async () => {
    // The server picks board_id over image_names (BulkDownloadService), so each chunk would
    // schedule the same full-board zip again: 1001 names plus a board id used to produce two
    // identical whole-board downloads.
    const baseQuery = vi.fn(
      (_args: Request): Promise<Response> => Promise.resolve({ data: { bulk_download_item_name: 'board.zip' } })
    );

    await run(baseQuery, { image_names: names(2500), board_id: 'board-1' });

    expect(baseQuery).toHaveBeenCalledTimes(1);
    expect(baseQuery.mock.calls[0]?.[0].body).toEqual({ image_names: undefined, board_id: 'board-1' });
  });

  it('stops scheduling zips when the session changes mid-run', async () => {
    // Switched on the *second* call, so a zip is already scheduled and `first` is set. Switching
    // on the first leaves `first` undefined, which makes returning it indistinguishable from
    // returning nothing -- and returning it is exactly what must not happen.
    login('user-a');
    let call = 0;
    const baseQuery = vi.fn((_args: Request): Promise<Response> => {
      call += 1;
      if (call === 2) {
        switchUser('user-b');
      }
      return Promise.resolve({ data: { bulk_download_item_name: `item-${call}.zip` } });
    });

    const result = await run(baseQuery, { image_names: names(2500) });

    expect(baseQuery).toHaveBeenCalledTimes(2);
    // Not `item-1.zip`: that zip belongs to the previous session. Do not expose its item name or
    // toast the new session about work it did not request.
    expect(result).toEqual({ data: undefined });
    expect(toast).not.toHaveBeenCalled();
    expect(i18n.t).not.toHaveBeenCalled();
  });

  it('reports an error when the first chunk fails, since nothing was scheduled', async () => {
    const baseQuery = vi.fn(
      (_args: Request): Promise<Response> => Promise.resolve({ error: { status: 403, data: 'nope' } })
    );

    const result = await run(baseQuery, { image_names: names(2500) });

    expect(result).toEqual({ error: { status: 403, data: 'nope' } });
    expect(baseQuery).toHaveBeenCalledTimes(1);
    // Nothing landed, so the failure toast belongs to the `matchRejected` listener alone.
    expect(toast).not.toHaveBeenCalled();
  });
});

describe('imageDTOsByNamesQueryFn', () => {
  type Request = { url: string; method: string; body: { image_names: string[] } };
  type Response = { data: ImageDTO[] } | { error: { status: number; data: string } };

  const dto = (image_name: string) => ({ image_name }) as ImageDTO;

  const run = (baseQuery: (args: Request) => Promise<Response>, image_names: string[]) => {
    const dispatch = vi.fn();
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const queryApi = { dispatch } as any;
    const query = baseQuery as any;
    /* eslint-enable @typescript-eslint/no-explicit-any */
    return { dispatch, result: imageDTOsByNamesQueryFn({ image_names }, queryApi, undefined, query) };
  };

  it('publishes each chunk as it arrives, so a later failure cannot discard the earlier ones', async () => {
    // This mutation rejects on any chunk failure and its only caller
    // (useRangeBasedImageFetching) never looks at the rejection -- so DTOs held back until the
    // end would be dropped for good. The hook re-requests only names missing from the cache,
    // and only when the user scrolls, so nothing would come back for them.
    let call = 0;
    const baseQuery = vi.fn((args: Request): Promise<Response> => {
      call += 1;
      if (call === 2) {
        return Promise.resolve({ error: { status: 500, data: 'boom' } });
      }
      return Promise.resolve({ data: args.body.image_names.map(dto) });
    });

    const { dispatch, result } = run(baseQuery, names(2500));

    expect(await result).toEqual({ error: { status: 500, data: 'boom' } });
    // One dispatch, carrying chunk one -- not zero, which is what holding the DTOs back until
    // the end would produce. Matched on the payload: the action also carries a request id and a
    // timestamp, both fresh per call.
    expect(dispatch).toHaveBeenCalledTimes(1);
    const action = dispatch.mock.calls[0]?.[0] as {
      type: string;
      payload: { value: ImageDTO }[];
    };
    expect(action.type).toBe(imagesApi.util.upsertQueryEntries([]).type);
    expect(action.payload.map((entry) => entry.value.image_name)).toEqual(names(1000));
  });

  it('checks the session with nothing between the check and the publish', () => {
    // Asserted structurally rather than by counting microtasks, which would break on any change
    // to the async shape rather than on the property. `fetchChunk` already checks the context
    // after the response, but resuming from it is a hop, and a logout landing in that hop passes
    // its check and still clears the cache before the upsert runs. Only a check with no await
    // between it and the write closes the window, so the write has to sit inside one.
    const source = readFileSync(fileURLToPath(new URL('./images.ts', import.meta.url)), 'utf8');
    expect(source).toMatch(/if \(isSameAuthContext\(authContext\)\) \{\s*upsertImageDTOs\(dispatch, chunkDTOs\);\s*\}/);
  });

  it('does not publish a chunk that came back after the session changed', async () => {
    // The DTOs were fetched as whoever was logged in when the chunk went out. By the time they
    // land the logout listener may have reset the api state for someone else, and writing them
    // in then seeds one user's cache with another's images.
    login('user-a');
    const baseQuery = vi.fn((args: Request): Promise<Response> => {
      switchUser('user-b');
      return Promise.resolve({ data: args.body.image_names.map(dto) });
    });

    const { dispatch, result } = run(baseQuery, names(2500));
    await result;

    expect(baseQuery).toHaveBeenCalledTimes(1);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('stops reading when the session changes mid-run', async () => {
    login('user-a');
    const baseQuery = vi.fn((args: Request): Promise<Response> => {
      switchUser('user-b');
      return Promise.resolve({ data: args.body.image_names.map(dto) });
    });

    const { result } = run(baseQuery, names(2500));

    expect(baseQuery).toHaveBeenCalledTimes(1);
    expect(await result).toEqual({ error: { status: 'CUSTOM_ERROR', error: expect.stringContaining('Aborted') } });
  });
});
