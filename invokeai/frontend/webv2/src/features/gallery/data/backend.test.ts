import { galleryItems } from '@features/gallery';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
  apiFetchRaw: vi.fn(),
  sleep: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  absolutizeApiUrl: (url: string) => `https://api.test${url}`,
  apiFetch: vi.fn(),
  apiFetchJson: mocks.apiFetchJson,
  apiFetchRaw: mocks.apiFetchRaw,
  sleep: mocks.sleep,
}));

import {
  addImagesToGalleryBoard,
  deleteGalleryBoard,
  deleteGalleryImages,
  downloadGalleryArchive,
  getGalleryImageByName,
  getGalleryImagesByNames,
  getGalleryVideoMetadata,
  getGalleryVideoWorkflow,
  imageMakeCanvasAssetChanges,
  imageMakeDurableChanges,
  imageSaveToGalleryChanges,
  listGalleryBoards,
  listGalleryItems,
  listPaletteImages,
  removeImagesFromGalleryBoard,
  starGalleryImages,
  unstarGalleryImages,
} from './backend';

describe('deleteGalleryBoard outcomes', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    mocks.apiFetchJson.mockReset();
  });

  it('maps confirmed and failed media when deleting a board with its contents', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      board_id: 'board-1',
      deleted_board_images: [],
      deleted_board_videos: [],
      deleted_images: ['gone.png'],
      deleted_videos: ['gone.mp4'],
      failed_images: ['locked.png'],
      failed_videos: ['locked.mp4'],
    });

    await expect(deleteGalleryBoard('board-1', true)).resolves.toEqual({
      boardId: 'board-1',
      deletedBoardImageNames: [],
      deletedBoardVideoNames: [],
      deletedImageNames: ['gone.png'],
      deletedVideoNames: ['gone.mp4'],
      failedImageNames: ['locked.png'],
      failedVideoNames: ['locked.mp4'],
    });
  });

  it('maps removed board relationships when retaining the board contents', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      board_id: 'board-1',
      deleted_board_images: ['moved.png'],
      deleted_board_videos: ['moved.mp4'],
      deleted_images: [],
      deleted_videos: [],
      failed_images: [],
      failed_videos: [],
    });

    await expect(deleteGalleryBoard('board-1', false)).resolves.toEqual({
      boardId: 'board-1',
      deletedBoardImageNames: ['moved.png'],
      deletedBoardVideoNames: ['moved.mp4'],
      deletedImageNames: [],
      deletedVideoNames: [],
      failedImageNames: [],
      failedVideoNames: [],
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/boards/board-1?include_images=false', {
      method: 'DELETE',
      signal: undefined,
    });
  });
});

describe('image mutation outcomes', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    mocks.apiFetchJson.mockReset();
  });

  it('returns only images confirmed added to or removed from a board', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce({ added_images: ['moved.png'], affected_boards: ['board-1', 'none'] })
      .mockResolvedValueOnce({ affected_boards: ['board-1', 'none'], removed_images: ['removed.png'] });

    await expect(addImagesToGalleryBoard('board-1', ['moved.png', 'locked.png'])).resolves.toEqual(['moved.png']);
    await expect(removeImagesFromGalleryBoard(['removed.png', 'locked.png'])).resolves.toEqual(['removed.png']);
  });

  it('returns only images confirmed starred or unstarred', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce({ affected_boards: ['board-1'], starred_images: ['starred.png'] })
      .mockResolvedValueOnce({ affected_boards: ['board-1'], unstarred_images: ['unstarred.png'] });

    await expect(starGalleryImages(['starred.png', 'locked.png'])).resolves.toEqual(['starred.png']);
    await expect(unstarGalleryImages(['unstarred.png', 'locked.png'])).resolves.toEqual(['unstarred.png']);
  });
});

describe('downloadGalleryArchive', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchRaw.mockReset();
    mocks.sleep.mockReset();
  });

  it('aborts polling before another account can request the prior account archive', async () => {
    let resumeSleep: (() => void) | undefined;
    mocks.apiFetchJson.mockResolvedValue({ bulk_download_item_name: 'user-a.zip' });
    mocks.apiFetchRaw.mockResolvedValue(new Response(null, { status: 404 }));
    mocks.sleep.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          resumeSleep = resolve;
        })
    );

    const download = downloadGalleryArchive({ imageNames: ['user-a.png'] });

    await vi.waitFor(() => {
      expect(mocks.sleep).toHaveBeenCalledOnce();
    });
    const postSignal = (mocks.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined)?.signal;
    const pollSignal = (mocks.apiFetchRaw.mock.calls[0]?.[1] as RequestInit | undefined)?.signal;
    const sleepSignal = mocks.sleep.mock.calls[0]?.[1] as AbortSignal;

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    resumeSleep?.();

    await expect(download).rejects.toThrow('no longer active');
    expect(postSignal?.aborted).toBe(true);
    expect(pollSignal?.aborted).toBe(true);
    expect(sleepSignal.aborted).toBe(true);
    expect(mocks.apiFetchRaw).toHaveBeenCalledTimes(1);
  });
});

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

describe('getGalleryImagesByNames', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('restores the requested order when the bulk endpoint responds out of order', async () => {
    const dto = (imageName: string) => ({
      board_id: 'none',
      created_at: '2026-07-09T12:00:00.000Z',
      height: 512,
      image_category: 'general',
      image_name: imageName,
      image_url: `/images/${imageName}`,
      is_intermediate: false,
      thumbnail_url: `/thumbnails/${imageName}`,
      width: 512,
    });
    mocks.apiFetchJson.mockResolvedValue([dto('third.png'), dto('first.png'), dto('second.png')]);

    const images = await getGalleryImagesByNames(['first.png', 'second.png', 'third.png']);

    expect(images.map((image) => image.imageName)).toEqual(['first.png', 'second.png', 'third.png']);
  });
});

describe('listPaletteImages created-at range', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ items: [], limit: 20, offset: 0, total: 0 });
  });

  it('sends created_from/created_to as query params', async () => {
    await listPaletteImages({
      boardId: 'board-1',
      createdFrom: '2026-07-01',
      createdTo: '2026-07-15',
      galleryView: 'images',
      searchTerm: '',
    });

    const url = mocks.apiFetchJson.mock.calls[0]?.[0] as string;
    const params = new URLSearchParams(url.split('?')[1]);
    expect(params.get('created_from')).toBe('2026-07-01');
    expect(params.get('created_to')).toBe('2026-07-15');
  });

  it('sends the explicit all-readable scope and forwards cancellation', async () => {
    const controller = new AbortController();

    await listPaletteImages({
      boardId: 'all',
      galleryView: 'images',
      searchTerm: 'sunset',
      signal: controller.signal,
    });

    const [url, init] = mocks.apiFetchJson.mock.calls[0] as [string, RequestInit];
    const params = new URLSearchParams(url.split('?')[1]);
    expect(params.get('board_id')).toBe('all');
    expect(init.signal).toBe(controller.signal);
  });

  it('omits the range params when no range is given', async () => {
    await listPaletteImages({ boardId: 'board-1', galleryView: 'images', searchTerm: 'sunset' });

    const url = mocks.apiFetchJson.mock.calls[0]?.[0] as string;
    const params = new URLSearchParams(url.split('?')[1]);
    expect(params.has('created_from')).toBe(false);
    expect(params.has('created_to')).toBe(false);
  });

  it('short-circuits a date board that falls outside the range without a request', async () => {
    await expect(
      listPaletteImages({
        boardId: 'by_date:2026-07-18',
        createdFrom: '2026-07-01',
        createdTo: '2026-07-15',
        galleryView: 'images',
        searchTerm: '',
      })
    ).resolves.toEqual({ images: [], total: 0 });
    expect(mocks.apiFetchJson).not.toHaveBeenCalled();
  });

  it('queries a date board normally when its day is inside the range', async () => {
    mocks.apiFetchJson.mockResolvedValue({ items: [], starred_count: 0, total_count: 0 });

    await expect(
      listPaletteImages({
        boardId: 'by_date:2026-07-10',
        createdFrom: '2026-07-01',
        createdTo: '2026-07-15',
        galleryView: 'images',
        searchTerm: '',
      })
    ).resolves.toEqual({ images: [], total: 0 });
    expect(mocks.apiFetchJson).toHaveBeenCalledOnce();
  });
});

describe('listGalleryItems', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('maps a mixed page by kind and normalizes transport-only fields', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      items: [
        {
          board_id: null,
          category: 'general',
          created_at: '2026-07-20T10:00:00.000Z',
          duration: 8.5,
          full_url: '/images/still.png',
          height: 512,
          is_intermediate: false,
          kind: 'image',
          name: 'still.png',
          starred: true,
          thumbnail_url: '/thumbnails/still.webp',
          width: 768,
        },
        {
          board_id: 'board-1',
          category: 'general',
          created_at: '2026-07-20T09:00:00.000Z',
          duration: 3.25,
          fps: null,
          full_url: '/videos/clip.mp4',
          height: 720,
          is_intermediate: true,
          kind: 'video',
          name: 'clip.mp4',
          starred: false,
          thumbnail_url: '/thumbnails/clip.webp',
          width: 1280,
        },
      ],
      limit: 20,
      offset: 0,
      total: 2,
    });

    await expect(listGalleryItems({ boardId: 'board-1', galleryView: 'images', searchTerm: '' })).resolves.toEqual({
      items: [
        {
          boardId: 'none',
          category: 'general',
          createdAt: '2026-07-20T10:00:00.000Z',
          fullUrl: 'https://api.test/images/still.png',
          height: 512,
          isIntermediate: false,
          kind: 'image',
          name: 'still.png',
          starred: true,
          thumbnailUrl: 'https://api.test/thumbnails/still.webp',
          width: 768,
        },
        {
          boardId: 'board-1',
          category: 'general',
          createdAt: '2026-07-20T09:00:00.000Z',
          durationSeconds: 3.25,
          fullUrl: 'https://api.test/videos/clip.mp4',
          height: 720,
          isIntermediate: true,
          kind: 'video',
          name: 'clip.mp4',
          starred: false,
          thumbnailUrl: 'https://api.test/thumbnails/clip.webp',
          width: 1280,
        },
      ],
      total: 2,
    });
  });

  it.each([undefined, null, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects a video whose duration is not finite: %s',
    async (duration) => {
      mocks.apiFetchJson.mockResolvedValue({
        items: [
          {
            board_id: null,
            category: 'general',
            created_at: '2026-07-20T09:00:00.000Z',
            duration,
            full_url: '/videos/broken.mp4',
            height: 720,
            is_intermediate: false,
            kind: 'video',
            name: 'broken.mp4',
            starred: false,
            thumbnail_url: '/thumbnails/broken.webp',
            width: 1280,
          },
        ],
        limit: 20,
        offset: 0,
        total: 1,
      });

      await expect(listGalleryItems({ boardId: 'none', galleryView: 'images', searchTerm: '' })).rejects.toThrow(
        'finite duration'
      );
    }
  );

  it('forwards gallery filters and pagination to the polymorphic endpoint', async () => {
    const controller = new AbortController();
    mocks.apiFetchJson.mockResolvedValue({ items: [], limit: 17, offset: 34, total: 0 });

    await listGalleryItems({
      boardId: 'board-1',
      createdFrom: '2026-07-01',
      createdTo: '2026-07-15',
      galleryView: 'images',
      isIntermediate: true,
      limit: 17,
      offset: 34,
      orderDir: 'ASC',
      searchTerm: '  summer clip  ',
      signal: controller.signal,
      starredFirst: true,
    });

    const [url, init] = mocks.apiFetchJson.mock.calls[0] as [string, RequestInit];
    const params = new URLSearchParams(url.split('?')[1]);

    expect(url.split('?')[0]).toBe('/api/v1/gallery/items/');
    expect(params.get('board_id')).toBe('board-1');
    expect(params.getAll('categories')).toEqual(['general']);
    expect(params.get('created_from')).toBe('2026-07-01');
    expect(params.get('created_to')).toBe('2026-07-15');
    expect(params.get('is_intermediate')).toBe('true');
    expect(params.get('limit')).toBe('17');
    expect(params.get('offset')).toBe('34');
    expect(params.get('order_dir')).toBe('ASC');
    expect(params.get('starred_first')).toBe('true');
    expect(params.get('search_term')).toBe('summer clip');
    expect(init.signal).toBe(controller.signal);
  });

  it('uses asset categories so general-category videos stay out of Assets', async () => {
    mocks.apiFetchJson.mockResolvedValue({ items: [], limit: 100, offset: 0, total: 0 });

    await listGalleryItems({ boardId: 'board-1', galleryView: 'assets', searchTerm: '' });

    const url = mocks.apiFetchJson.mock.calls[0]?.[0] as string;
    const categories = new URLSearchParams(url.split('?')[1]).getAll('categories');

    expect(categories).toEqual(['control', 'mask', 'user']);
    expect(categories).not.toContain('general');
  });
});

describe('palette image search', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ items: [], limit: 20, offset: 0, total: 0 });
  });

  it('keeps all-readable palette searches on the image-only endpoint', async () => {
    await listPaletteImages({ boardId: 'all', galleryView: 'images', limit: 20, searchTerm: '  portrait  ' });

    const url = mocks.apiFetchJson.mock.calls[0]?.[0] as string;
    const params = new URLSearchParams(url.split('?')[1]);

    expect(url.split('?')[0]).toBe('/api/v1/images/');
    expect(params.get('board_id')).toBe('all');
    expect(params.get('search_term')).toBe('portrait');
  });
});

describe('mixed gallery item details', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('resolves an image ref through the strict image DTO mapper', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      board_id: null,
      created_at: '2026-07-20T10:00:00.000Z',
      height: 512,
      image_category: 'general',
      image_name: 'folder/still.png',
      image_url: '/images/still.png',
      is_intermediate: false,
      starred: false,
      thumbnail_url: '/thumbnails/still.webp',
      width: 768,
    });

    await expect(galleryItems.resolve({ kind: 'image', name: 'folder/still.png' })).resolves.toMatchObject({
      kind: 'image',
      name: 'folder/still.png',
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/images/i/folder%2Fstill.png', { signal: undefined });
  });

  it('resolves a video ref and omits absent fps', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      board_id: null,
      created_at: '2026-07-20T10:00:00.000Z',
      duration: 4.5,
      fps: null,
      height: 720,
      is_intermediate: false,
      starred: true,
      thumbnail_url: '/thumbnails/clip.webp',
      video_category: 'general',
      video_name: 'folder/clip.mp4',
      video_url: '/videos/clip.mp4',
      width: 1280,
    });

    const item = await galleryItems.resolve({ kind: 'video', name: 'folder/clip.mp4' });

    expect(item).toEqual({
      boardId: 'none',
      category: 'general',
      createdAt: '2026-07-20T10:00:00.000Z',
      durationSeconds: 4.5,
      fullUrl: 'https://api.test/videos/clip.mp4',
      height: 720,
      isIntermediate: false,
      kind: 'video',
      name: 'folder/clip.mp4',
      starred: true,
      thumbnailUrl: 'https://api.test/thumbnails/clip.webp',
      width: 1280,
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/videos/i/folder%2Fclip.mp4', { signal: undefined });
  });

  it('fetches focused video metadata and workflow details', async () => {
    const controller = new AbortController();
    mocks.apiFetchJson
      .mockResolvedValueOnce({ positive_prompt: 'moving portrait' })
      .mockResolvedValueOnce({ graph: '{"nodes":[]}', workflow: null });

    await expect(getGalleryVideoMetadata('folder/clip.mp4', controller.signal)).resolves.toEqual({
      positive_prompt: 'moving portrait',
    });
    await expect(getGalleryVideoWorkflow('folder/clip.mp4', controller.signal)).resolves.toEqual({
      graph: '{"nodes":[]}',
      workflow: null,
    });
    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(1, '/api/v1/videos/i/folder%2Fclip.mp4/metadata', {
      signal: controller.signal,
    });
    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(2, '/api/v1/videos/i/folder%2Fclip.mp4/workflow', {
      signal: controller.signal,
    });
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

describe('imageMakeCanvasAssetChanges', () => {
  it('makes the image durable AND moves it out of the gallery images view', () => {
    // A node's output is `general` — precisely what the Images view lists — so
    // durability alone published every ControlNet preprocess into the gallery.
    expect(imageMakeCanvasAssetChanges()).toEqual({ image_category: 'other', is_intermediate: false });
  });

  it('uses the same category the canvas uploads its own paint bitmaps under', () => {
    // Layer pixels are layer pixels however they were produced; a filtered
    // control map must not be classified differently from a painted one.
    expect(imageMakeCanvasAssetChanges().image_category).toBe('other');
  });

  it('still clears is_intermediate, so the layer source cannot be collected', () => {
    expect(imageMakeCanvasAssetChanges().is_intermediate).toBe(false);
  });
});

describe('gallery category queries', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ items: [], limit: 20, offset: 0, total: 0 });
  });

  const categoriesFor = async (galleryView: 'images' | 'assets'): Promise<string[]> => {
    await listPaletteImages({ boardId: 'board-1', galleryView, searchTerm: '' });
    const url = mocks.apiFetchJson.mock.calls[0]?.[0] as string;

    return new URLSearchParams(url.split('?')[1]).getAll('categories');
  };

  it('never asks the assets view for the canvas-owned category', async () => {
    // The regression this guards: canvas paint bitmaps, composites and adopted
    // filter results all upload as `other`. While `other` was an assets
    // category, every brush stroke and every generation put an image in the
    // user's Assets tab.
    expect(await categoriesFor('assets')).not.toContain('other');
  });

  it('lists only user-facing categories in the assets view', async () => {
    expect(await categoriesFor('assets')).toEqual(['control', 'mask', 'user']);
  });

  it('lists only general in the images view', async () => {
    expect(await categoriesFor('images')).toEqual(['general']);
  });

  it('leaves the canvas category out of both views', async () => {
    const assets = await categoriesFor('assets');

    mocks.apiFetchJson.mockClear();
    const images = await categoriesFor('images');

    expect([...assets, ...images]).not.toContain('other');
  });
});

describe('board media counts and covers', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    mocks.apiFetchJson.mockReset();
  });

  it('maps video counts and resolves a video cover to its thumbnail', async () => {
    mocks.apiFetchJson.mockImplementation((path: string) => {
      if (path.startsWith('/api/v1/boards/')) {
        return Promise.resolve([
          {
            archived: false,
            asset_count: 1,
            board_id: 'board-1',
            board_name: 'Clips',
            cover_image_name: null,
            cover_video_name: 'clip.mp4',
            image_count: 2,
            video_count: 3,
          },
        ]);
      }

      // Uncategorized image / asset / video totals.
      return Promise.resolve({ total: 0 });
    });

    const boards = await listGalleryBoards();

    expect(boards[1]).toMatchObject({
      coverThumbnailUrl: 'https://api.test/api/v1/videos/i/clip.mp4/thumbnail',
      coverVideoName: 'clip.mp4',
      id: 'board-1',
      imageCount: 2,
      videoCount: 3,
    });
  });

  it('counts uncategorized videos, which no board DTO reports', async () => {
    mocks.apiFetchJson.mockImplementation((path: string) => {
      if (path.startsWith('/api/v1/boards/')) {
        return Promise.resolve([]);
      }

      if (path.startsWith('/api/v1/videos/')) {
        return Promise.resolve({ total: 7 });
      }

      return Promise.resolve({ total: 4 });
    });

    const boards = await listGalleryBoards();

    expect(boards[0]).toMatchObject({ id: 'none', kind: 'uncategorized', videoCount: 7 });
  });

  it('treats a board with only videos as non-empty even though image_count is 0', async () => {
    mocks.apiFetchJson.mockImplementation((path: string) => {
      if (path.startsWith('/api/v1/boards/')) {
        return Promise.resolve([
          {
            archived: false,
            asset_count: 0,
            board_id: 'video-only',
            board_name: 'Video only',
            cover_image_name: null,
            cover_video_name: 'only.mp4',
            image_count: 0,
            video_count: 5,
          },
        ]);
      }

      return Promise.resolve({ total: 0 });
    });

    const boards = await listGalleryBoards();

    expect(boards[1]).toMatchObject({ imageCount: 0, videoCount: 5 });
    expect(boards[1]?.coverThumbnailUrl).toBeDefined();
  });
});

describe('deleteGalleryImages partial failures', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    mocks.apiFetchJson.mockReset();
  });

  it('separates deleted names from ones the backend refused', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      affected_boards: ['board-1'],
      deleted_images: ['gone.png'],
      failed_images: ['locked.png'],
    });

    await expect(deleteGalleryImages(['gone.png', 'locked.png'])).resolves.toEqual({
      deletedImageNames: ['gone.png'],
      failedImageNames: ['locked.png'],
    });
  });

  it('reports no failures when the backend omits failed_images', async () => {
    mocks.apiFetchJson.mockResolvedValue({ affected_boards: [], deleted_images: ['gone.png'] });

    await expect(deleteGalleryImages(['gone.png'])).resolves.toEqual({
      deletedImageNames: ['gone.png'],
      failedImageNames: [],
    });
  });

  it('skips the request entirely for an empty selection', async () => {
    await expect(deleteGalleryImages([])).resolves.toEqual({ deletedImageNames: [], failedImageNames: [] });
    expect(mocks.apiFetchJson).not.toHaveBeenCalled();
  });
});
