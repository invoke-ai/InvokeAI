import { ApiError } from '@platform/transport/http';
import { describe, expect, it, vi } from 'vitest';

import type * as assetTransportModule from './assetTransport';

import { INVK_MAX_ARCHIVE_BYTES } from './archive';

const mocks = vi.hoisted(() => ({ apiFetchJson: vi.fn(), apiFetchRaw: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  apiFetchJson: mocks.apiFetchJson,
  apiFetchRaw: mocks.apiFetchRaw,
}));

import {
  createAssetResponseReader,
  fetchImageBytes,
  fetchImageThumbnail,
  fetchVideoBytes,
  findExistingVideoNames,
} from './assetTransport';

describe('findExistingVideoNames', () => {
  it('returns a video name when its lookup succeeds', async () => {
    mocks.apiFetchRaw.mockResolvedValue(new Response(null, { status: 200 }));

    await expect(findExistingVideoNames(['present.mp4'])).resolves.toEqual(new Set(['present.mp4']));
  });

  it.each([403, 404])('treats a %i video lookup as absent', async (status) => {
    mocks.apiFetchRaw.mockResolvedValue(new Response(null, { status }));

    await expect(findExistingVideoNames(['missing.mp4'])).resolves.toEqual(new Set());
  });

  it.each([401, 422, 500])('rejects a %i video lookup with an ApiError', async (status) => {
    mocks.apiFetchRaw.mockResolvedValue(new Response('lookup failed', { status }));

    await expect(findExistingVideoNames(['failed.mp4'])).rejects.toBeInstanceOf(ApiError);
  });

  it('propagates a network rejection unchanged', async () => {
    const networkError = new Error('network unavailable');
    mocks.apiFetchRaw.mockRejectedValue(networkError);

    await expect(findExistingVideoNames(['unreachable.mp4'])).rejects.toBe(networkError);
  });
});

describe('import rollback transport', () => {
  it('sends authoritative image and video identities to their batch-delete routes', async () => {
    const transport = (await import('./assetTransport')) as typeof assetTransportModule & {
      deleteArchiveImages?: (names: string[], signal?: AbortSignal) => Promise<void>;
      deleteArchiveVideos?: (names: string[], signal?: AbortSignal) => Promise<void>;
    };

    expect(transport.deleteArchiveImages).toBeTypeOf('function');
    expect(transport.deleteArchiveVideos).toBeTypeOf('function');

    if (!transport.deleteArchiveImages || !transport.deleteArchiveVideos) {
      return;
    }

    const signal = new AbortController().signal;

    mocks.apiFetchJson.mockResolvedValue({});
    await transport.deleteArchiveImages(['server-image.png'], signal);
    await transport.deleteArchiveVideos(['server-video.mp4'], signal);

    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(1, '/api/v1/images/delete', {
      body: JSON.stringify({ image_names: ['server-image.png'] }),
      method: 'POST',
      signal,
    });
    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(2, '/api/v1/videos/delete', {
      body: JSON.stringify({ video_names: ['server-video.mp4'] }),
      method: 'POST',
      signal,
    });
  });
});

describe('export asset response limits', () => {
  it('rejects an oversized Content-Length before materializing the body', async () => {
    const response = new Response(null, {
      headers: { 'content-length': String(INVK_MAX_ARCHIVE_BYTES + 1) },
      status: 200,
    });
    const arrayBuffer = vi.spyOn(response, 'arrayBuffer');

    mocks.apiFetchRaw.mockResolvedValue(response);

    await expect(fetchImageBytes('too-large.png')).rejects.toMatchObject({ reason: 'too-large' });
    expect(arrayBuffer).not.toHaveBeenCalled();
  });

  it('rejects and cancels a stream that crosses the per-response ceiling', async () => {
    const cancel = vi.fn();
    const atLimit = { byteLength: INVK_MAX_ARCHIVE_BYTES } as Uint8Array;
    const oneByte = { byteLength: 1 } as Uint8Array;
    let chunk = 0;
    const response = new Response(
      new ReadableStream<Uint8Array>({
        cancel,
        pull(controller) {
          controller.enqueue(chunk === 0 ? atLimit : oneByte);
          chunk += 1;
        },
      }),
      { status: 200 }
    );
    const arrayBuffer = vi.spyOn(response, 'arrayBuffer');

    mocks.apiFetchRaw.mockResolvedValue(response);

    await expect(fetchVideoBytes('too-large.mp4')).rejects.toMatchObject({ reason: 'too-large' });
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(arrayBuffer).not.toHaveBeenCalled();
  });

  it('streams thumbnail chunks into the returned bytes', async () => {
    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2]));
          controller.enqueue(new Uint8Array([3, 4]));
          controller.close();
        },
      }),
      { headers: { 'content-type': 'image/png' }, status: 200 }
    );
    const arrayBuffer = vi.spyOn(response, 'arrayBuffer');

    mocks.apiFetchRaw.mockResolvedValue(response);

    await expect(fetchImageThumbnail('cover.png')).resolves.toEqual({
      bytes: new Uint8Array([1, 2, 3, 4]),
      contentType: 'image/png',
    });
    expect(arrayBuffer).not.toHaveBeenCalled();
  });

  it('accepts the cumulative boundary and refuses the next declared byte before reading it', async () => {
    const readResponse = createAssetResponseReader();
    const atLimit = { byteLength: INVK_MAX_ARCHIVE_BYTES } as Uint8Array;
    const firstResponse = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(atLimit);
          controller.close();
        },
      })
    );

    await expect(readResponse(firstResponse)).resolves.toBe(atLimit);

    const cancel = vi.fn(() => Promise.resolve());
    const getReader = vi.fn(() => ({
      cancel,
      read: vi
        .fn()
        .mockResolvedValueOnce({ done: false, value: new Uint8Array([1]) })
        .mockResolvedValueOnce({ done: true, value: undefined }),
      releaseLock: vi.fn(),
    }));
    const nextResponse = {
      body: { cancel, getReader },
      headers: new Headers({ 'content-length': '1' }),
    } as unknown as Response;

    await expect(readResponse(nextResponse)).rejects.toMatchObject({ reason: 'too-large' });
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(getReader).not.toHaveBeenCalled();
  });

  it('cancels a pending reader as soon as its signal is aborted', async () => {
    let settleRead: (result: ReadableStreamReadResult<Uint8Array>) => void = () => undefined;
    const pendingRead = new Promise<ReadableStreamReadResult<Uint8Array>>((resolve) => {
      settleRead = resolve;
    });
    const cancel = vi.fn(() => {
      settleRead({ done: true, value: undefined });

      return Promise.resolve();
    });
    const reader = {
      cancel,
      read: vi.fn(() => pendingRead),
      releaseLock: vi.fn(),
    };
    const response = {
      body: { cancel: vi.fn(() => Promise.resolve()), getReader: vi.fn(() => reader) },
      headers: new Headers(),
    } as unknown as Response;
    const controller = new AbortController();
    const read = createAssetResponseReader()(response, controller.signal);

    controller.abort(new DOMException('The operation was aborted.', 'AbortError'));
    await Promise.resolve();

    const cancellationCallsWhileReadWasPending = cancel.mock.calls.length;

    if (cancellationCallsWhileReadWasPending === 0) {
      settleRead({ done: true, value: undefined });
    }

    await expect(read).rejects.toBe(controller.signal.reason);
    expect(cancellationCallsWhileReadWasPending).toBe(1);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(reader.releaseLock).toHaveBeenCalledTimes(1);
  });
});
