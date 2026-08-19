import { toast } from 'features/toast/toast';
import i18n from 'i18next';
import {
  buildChunkedImageBatchQueryFn,
  bulkDownloadQueryFn,
  chunkImageNames,
  mergeImageBatchResults,
} from 'services/api/endpoints/images';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '..';

vi.mock('features/toast/toast', () => ({ toast: vi.fn() }));
vi.mock('i18next', () => ({ default: { t: vi.fn((key: string) => key) } }));

/** Mirrors MAX_IMAGE_BATCH_SIZE in invokeai/app/api/routers/images.py. */
const CHUNK_SIZE = 1000;

const names = (count: number) => Array.from({ length: count }, (_, i) => `image-${i}.png`);

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
