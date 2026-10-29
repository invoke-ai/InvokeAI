import { afterEach, describe, expect, it, vi } from 'vitest';

import { CanvasImageUploadError, uploadCanvasImage } from './canvasImages';

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(typeof body === 'string' ? body : JSON.stringify(body), { status });

const stubFetch = (implementation: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) => {
  const fetchMock = vi.fn(implementation);

  vi.stubGlobal('fetch', fetchMock);

  return fetchMock;
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('uploadCanvasImage', () => {
  it('forwards an abort signal to fetch', async () => {
    const controller = new AbortController();
    const fetchMock = stubFetch(() =>
      Promise.resolve(jsonResponse({ height: 1, image_name: 'uploaded.png', width: 1 }))
    );

    await uploadCanvasImage(new Blob(['pixels'], { type: 'image/png' }), { signal: controller.signal });

    const signal = fetchMock.mock.calls[0]?.[1]?.signal;

    expect(signal).toBeInstanceOf(AbortSignal);
    expect(signal?.aborted).toBe(false);
    controller.abort();
    expect(signal?.aborted).toBe(true);
  });

  it('POSTs a multipart file to the upload endpoint with the persistence params', async () => {
    const fetchMock = stubFetch(() =>
      Promise.resolve(jsonResponse({ height: 64, image_name: 'paint-1.png', width: 128 }, 201))
    );
    const blob = new Blob(['png-bytes'], { type: 'image/png' });

    const result = await uploadCanvasImage(blob);

    expect(result).toEqual({ height: 64, imageName: 'paint-1.png', width: 128 });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/api/v1/images/upload');
    expect(url).toContain('image_category=other');
    expect(url).toContain('is_intermediate=false');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    const file = (init.body as FormData).get('file');
    expect(file).toBeInstanceOf(File);
    expect((file as File).type).toBe('image/png');
  });

  it('honors overrides for category, intermediate flag, board, metadata, and resize dimensions', async () => {
    const fetchMock = stubFetch(() => Promise.resolve(jsonResponse({ height: 1, image_name: 'x', width: 1 }, 201)));

    await uploadCanvasImage(new Blob([''], { type: 'image/png' }), {
      boardId: 'board-9',
      imageCategory: 'user',
      isIntermediate: true,
      metadata: { seed: 42 },
      resizeTo: { width: 1024, height: 768 },
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('image_category=user');
    expect(url).toContain('is_intermediate=true');
    expect(url).toContain('board_id=board-9');
    const formData = init.body as FormData;
    expect(formData.get('metadata')).toBe(JSON.stringify({ seed: 42 }));
    expect(formData.get('resize_to')).toBe(JSON.stringify({ width: 1024, height: 768 }));
  });

  it('throws a typed error with the status on a non-ok response', async () => {
    stubFetch(() => Promise.resolve(jsonResponse('Not an image', 415)));

    await expect(uploadCanvasImage(new Blob([''], { type: 'image/png' }))).rejects.toMatchObject({
      name: 'CanvasImageUploadError',
      status: 415,
    });
  });

  it('wraps a network failure in a CanvasImageUploadError', async () => {
    stubFetch(() => Promise.reject(new Error('network down')));

    const error = await uploadCanvasImage(new Blob([''], { type: 'image/png' })).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(CanvasImageUploadError);
    expect((error as CanvasImageUploadError).status).toBeNull();
    expect((error as CanvasImageUploadError).message).toContain('network down');
  });
});
