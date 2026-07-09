import { describe, expect, it, vi } from 'vitest';

import { CanvasImageUploadError, uploadCanvasImage } from './canvasImages';

/** A minimal `Response`-shaped fake for the injected fetch. */
const jsonResponse = (body: unknown, init: { ok?: boolean; status?: number; statusText?: string } = {}): Response =>
  ({
    json: () => Promise.resolve(body),
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
  }) as unknown as Response;

describe('uploadCanvasImage', () => {
  it('POSTs a multipart file to the upload endpoint with the persistence params', async () => {
    const fetchImpl = vi.fn(() =>
      Promise.resolve(jsonResponse({ height: 64, image_name: 'paint-1.png', width: 128 }, { status: 201 }))
    );
    const blob = new Blob(['png-bytes'], { type: 'image/png' });

    const result = await uploadCanvasImage(blob, { fetch: fetchImpl });

    expect(result).toEqual({ height: 64, imageName: 'paint-1.png', width: 128 });
    expect(fetchImpl).toHaveBeenCalledTimes(1);

    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toContain('/api/v1/images/upload');
    expect(url).toContain('image_category=other');
    expect(url).toContain('is_intermediate=false');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    const file = (init.body as FormData).get('file');
    expect(file).toBeInstanceOf(File);
    expect((file as File).type).toBe('image/png');
  });

  it('honors overrides for category, intermediate flag, and board', async () => {
    const fetchImpl = vi.fn(() =>
      Promise.resolve(jsonResponse({ height: 1, image_name: 'x', width: 1 }, { status: 201 }))
    );

    await uploadCanvasImage(new Blob([''], { type: 'image/png' }), {
      boardId: 'board-9',
      fetch: fetchImpl,
      imageCategory: 'user',
      isIntermediate: true,
    });

    const [url] = fetchImpl.mock.calls[0] as unknown as [string];
    expect(url).toContain('image_category=user');
    expect(url).toContain('is_intermediate=true');
    expect(url).toContain('board_id=board-9');
  });

  it('throws a typed error with the status on a non-ok response', async () => {
    const fetchImpl = vi.fn(() =>
      Promise.resolve(jsonResponse('Not an image', { ok: false, status: 415, statusText: 'Unsupported Media Type' }))
    );

    await expect(uploadCanvasImage(new Blob([''], { type: 'image/png' }), { fetch: fetchImpl })).rejects.toMatchObject({
      name: 'CanvasImageUploadError',
      status: 415,
    });
  });

  it('wraps a network failure in a CanvasImageUploadError', async () => {
    const fetchImpl = vi.fn(() => Promise.reject(new Error('network down')));

    const error = await uploadCanvasImage(new Blob([''], { type: 'image/png' }), { fetch: fetchImpl }).catch(
      (caught: unknown) => caught
    );

    expect(error).toBeInstanceOf(CanvasImageUploadError);
    expect((error as CanvasImageUploadError).status).toBeNull();
    expect((error as CanvasImageUploadError).message).toContain('network down');
  });
});
