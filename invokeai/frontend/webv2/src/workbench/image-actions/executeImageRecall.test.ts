import type { GalleryImage } from '@workbench/gallery/api';
import type { ModelConfig } from '@workbench/models/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  getGalleryImageMetadata: vi.fn(),
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
      })
    ).resolves.toBe(true);

    expect(galleryApi.getGalleryImageMetadata).toHaveBeenCalledWith('selected.png');
    expect(dispatch).toHaveBeenCalledWith({
      type: 'setGenerateSettings',
      values: expect.objectContaining({ positivePrompt: 'recalled prompt' }),
    });
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'success', title: 'Recalled remix settings', type: 'recordNotice' })
    );
  });
});
