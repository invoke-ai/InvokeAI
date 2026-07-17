import type { GalleryImage } from '@workbench/gallery/api';
import type { ModelConfig } from '@workbench/models/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  getGalleryImageMetadata: vi.fn(),
  getGalleryImagesByNames: vi.fn(),
}));

vi.mock('@workbench/gallery/api', () => galleryApi);

import { executeImageRecall } from './executeImageRecall';

const model = {
  base: 'sdxl',
  file_size: 1,
  format: 'checkpoint',
  hash: 'hash',
  key: 'sdxl-model',
  name: 'SDXL',
  path: '/models/sdxl.safetensors',
  source: 'local',
  source_type: 'path',
  type: 'main',
} as ModelConfig;

const image: GalleryImage = {
  boardId: 'none',
  height: 768,
  imageCategory: 'general',
  imageName: 'selected.png',
  imageUrl: '/selected.png',
  queuedAt: '2026-06-19T00:00:00.000Z',
  sourceQueueItemId: 'backend-gallery',
  starred: false,
  thumbnailUrl: '/selected-thumb.png',
  width: 512,
};

describe('executeImageRecall', () => {
  beforeEach(() => {
    galleryApi.getGalleryImageMetadata.mockReset();
    galleryApi.getGalleryImagesByNames.mockReset();
  });

  it('recalls remix settings into Generate values', async () => {
    const dispatch = vi.fn();

    galleryApi.getGalleryImageMetadata.mockResolvedValue({ positive_prompt: 'recalled prompt' });

    await expect(
      executeImageRecall({
        dispatch,
        generateValues: { modelKey: model.key },
        image,
        kind: 'remix',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(galleryApi.getGalleryImageMetadata).toHaveBeenCalledWith('selected.png');
    expect(dispatch).toHaveBeenCalledWith({
      projectId: 'project-1',
      type: 'setGenerateSettings',
      values: expect.objectContaining({ positivePrompt: 'recalled prompt' }),
    });
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'success', title: 'Recalled remix settings', type: 'recordNotice' })
    );
  });

  it('uses the freshest Generate values when recalling image dimensions', async () => {
    const dispatch = vi.fn();

    await expect(
      executeImageRecall({
        dispatch,
        generateValues: { modelKey: model.key, positivePrompt: 'stale prompt' },
        getGenerateValues: () => ({ modelKey: model.key, positivePrompt: 'fresh prompt' }),
        image,
        kind: 'dimensions',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(dispatch).toHaveBeenCalledWith({
      projectId: 'project-1',
      type: 'setGenerateSettings',
      values: expect.objectContaining({ positivePrompt: 'fresh prompt', height: image.height, width: image.width }),
    });
  });

  it('bulk-checks effective reference images and omits deleted entries before committing', async () => {
    const dispatch = vi.fn();
    const recallImage = { ...image, imageName: 'selected-with-references.png' };
    const makeReference = (id: string, imageName: string) => ({
      config: {
        image: {
          crop: {
            box: { height: 128, width: 128, x: 0, y: 0 },
            image: { height: 128, image_name: imageName, width: 128 },
            ratio: 1,
          },
          original: { image: { height: 256, image_name: `${id}-original.png`, width: 256 } },
        },
        type: 'qwen_image_reference_image',
      },
      id,
      isEnabled: true,
    });

    galleryApi.getGalleryImageMetadata.mockResolvedValue({
      positive_prompt: 'keep other fields',
      ref_images: [makeReference('valid', 'valid-crop.png'), makeReference('deleted', 'deleted-crop.png')],
    });
    galleryApi.getGalleryImagesByNames.mockResolvedValue([{ ...image, imageName: 'valid-crop.png' }]);

    await expect(
      executeImageRecall({
        dispatch,
        generateValues: { modelKey: model.key },
        image: recallImage,
        kind: 'all',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(galleryApi.getGalleryImagesByNames).toHaveBeenCalledWith(['valid-crop.png', 'deleted-crop.png']);
    expect(dispatch).toHaveBeenCalledWith({
      projectId: 'project-1',
      type: 'setGenerateSettings',
      values: expect.objectContaining({
        positivePrompt: 'keep other fields',
        referenceImages: [expect.objectContaining({ id: 'valid' })],
      }),
    });
  });
});
