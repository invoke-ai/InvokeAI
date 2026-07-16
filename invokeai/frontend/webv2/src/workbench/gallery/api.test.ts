import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@workbench/backend/http', () => ({
  absolutizeApiUrl: (url: string) => `https://api.test${url}`,
  apiFetch: vi.fn(),
  apiFetchJson: mocks.apiFetchJson,
  apiFetchRaw: vi.fn(),
  sleep: vi.fn(),
}));

import { getGalleryImageByName, imageMakeDurableChanges, imageSaveToGalleryChanges } from './api';

describe('getGalleryImageByName', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('encodes the image name, forwards the abort signal, and maps the backend image', async () => {
    const controller = new AbortController();
    mocks.apiFetchJson.mockResolvedValue({
      board_id: 'board-1',
      created_at: '2026-07-09T12:00:00.000Z',
      height: 360,
      image_category: 'general',
      image_name: 'folder/workflow result.png',
      image_url: '/images/workflow-result.png',
      is_intermediate: false,
      starred: true,
      thumbnail_url: '/thumbnails/workflow-result.webp',
      width: 640,
    });

    await expect(getGalleryImageByName('folder/workflow result.png', controller.signal)).resolves.toEqual({
      boardId: 'board-1',
      height: 360,
      imageCategory: 'general',
      imageName: 'folder/workflow result.png',
      imageUrl: 'https://api.test/images/workflow-result.png',
      queuedAt: '2026-07-09T12:00:00.000Z',
      sourceQueueItemId: 'backend-gallery',
      starred: true,
      thumbnailUrl: 'https://api.test/thumbnails/workflow-result.webp',
      width: 640,
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/images/i/folder%2Fworkflow%20result.png', {
      signal: controller.signal,
    });
  });

  it('propagates backend lookup errors', async () => {
    const error = new Error('404 image not found');
    mocks.apiFetchJson.mockRejectedValue(error);

    await expect(getGalleryImageByName('missing.png')).rejects.toBe(error);
  });

  it('propagates request aborts', async () => {
    const controller = new AbortController();
    const error = new DOMException('The operation was aborted.', 'AbortError');
    mocks.apiFetchJson.mockRejectedValue(error);

    const result = getGalleryImageByName('workflow.png', controller.signal);
    controller.abort();

    await expect(result).rejects.toBe(error);
  });
});

describe('imageSaveToGalleryChanges', () => {
  it('promotes an intermediate candidate to a durable, gallery-visible image', () => {
    // The backend `ImageRecordChanges` body: clear `is_intermediate` (stop GC)
    // and set `image_category: 'general'` (surface in the gallery images view).
    expect(imageSaveToGalleryChanges()).toEqual({ image_category: 'general', is_intermediate: false });
  });
});

describe('imageMakeDurableChanges', () => {
  it('clears is_intermediate without changing the existing image category', () => {
    // A utility result adopted by the canvas must survive GC, while its existing
    // category continues to determine whether it appears in Images or Assets.
    const body = imageMakeDurableChanges();
    expect(body).toEqual({ is_intermediate: false });
    expect('image_category' in body).toBe(false);
  });
});
