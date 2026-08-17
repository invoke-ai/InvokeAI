import { toast } from 'features/toast/toast';
import { buildChunkedImageBatchQueryFn, chunkImageNames, mergeImageBatchResults } from 'services/api/endpoints/images';
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
