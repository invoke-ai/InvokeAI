import type { GalleryImage } from '@features/gallery';
import type { ModelConfig } from '@features/models';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryImages: {
    metadata: vi.fn(),
    resolveMany: vi.fn(),
  },
}));

vi.mock('@features/gallery', () => galleryApi);

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

const createCommands = () => {
  const add = vi.fn();
  const setSettings = vi.fn();
  const commands = {
    generation: { setSettings } as unknown as WorkbenchCommands['generation'],
    notifications: { add, reportError: vi.fn() } as unknown as WorkbenchCommands['notifications'],
  };

  return { add, commands, setSettings };
};

describe('executeImageRecall', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    galleryApi.galleryImages.metadata.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
  });

  it('recalls remix settings into Generate values', async () => {
    const { add, commands, setSettings } = createCommands();

    galleryApi.galleryImages.metadata.mockResolvedValue({ positive_prompt: 'recalled prompt' });

    await expect(
      executeImageRecall({
        commands,
        generateValues: { modelKey: model.key },
        image,
        kind: 'remix',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(galleryApi.galleryImages.metadata).toHaveBeenCalledWith('selected.png', expect.any(AbortSignal));
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'recalled prompt' }),
      'project-1'
    );
    expect(add).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', title: 'Recalled remix settings' }));
  });

  it('uses the freshest Generate values when recalling image dimensions', async () => {
    const { commands, setSettings } = createCommands();

    await expect(
      executeImageRecall({
        commands,
        generateValues: { modelKey: model.key, positivePrompt: 'stale prompt' },
        getGenerateValues: () => ({ modelKey: model.key, positivePrompt: 'fresh prompt' }),
        image,
        kind: 'dimensions',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'fresh prompt', height: image.height, width: image.width }),
      'project-1'
    );
  });

  it('bulk-checks effective reference images and omits deleted entries before committing', async () => {
    const { commands, setSettings } = createCommands();
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

    galleryApi.galleryImages.metadata.mockResolvedValue({
      positive_prompt: 'keep other fields',
      ref_images: [makeReference('valid', 'valid-crop.png'), makeReference('deleted', 'deleted-crop.png')],
    });
    galleryApi.galleryImages.resolveMany.mockResolvedValue([{ ...image, imageName: 'valid-crop.png' }]);

    await expect(
      executeImageRecall({
        commands,
        generateValues: { modelKey: model.key },
        image: recallImage,
        kind: 'all',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(galleryApi.galleryImages.resolveMany).toHaveBeenCalledWith(
      ['valid-crop.png', 'deleted-crop.png'],
      expect.any(AbortSignal)
    );
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        positivePrompt: 'keep other fields',
        referenceImages: [expect.objectContaining({ id: 'valid' })],
      }),
      'project-1'
    );
  });

  it('does not reuse or commit image metadata from an expired account epoch', async () => {
    const oldCommands = createCommands();
    let resolveOldMetadata: ((value: unknown) => void) | undefined;

    galleryApi.galleryImages.metadata.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveOldMetadata = resolve;
      })
    );
    const oldRecall = executeImageRecall({
      commands: oldCommands.commands,
      generateValues: { modelKey: model.key },
      image,
      kind: 'remix',
      models: [model],
      projectId: 'project-a',
    });

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');

    const newCommands = createCommands();
    galleryApi.galleryImages.metadata.mockResolvedValueOnce({ positive_prompt: 'user b' });
    await expect(
      executeImageRecall({
        commands: newCommands.commands,
        generateValues: { modelKey: model.key },
        image,
        kind: 'remix',
        models: [model],
        projectId: 'project-b',
      })
    ).resolves.toBe(true);

    resolveOldMetadata?.({ positive_prompt: 'user a' });
    await expect(oldRecall).resolves.toBe(false);

    expect(oldCommands.setSettings).not.toHaveBeenCalled();
    expect(newCommands.setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'user b' }),
      'project-b'
    );
    expect(galleryApi.galleryImages.metadata).toHaveBeenCalledTimes(2);
  });

  it('does not recapture the current account when a caller-owned scope has already expired', async () => {
    const oldCommands = createCommands();
    const owner = accountLifecycle.capture();

    accountLifecycle.activate('new-account');

    await expect(
      executeImageRecall({
        commands: oldCommands.commands,
        generateValues: { modelKey: model.key },
        image,
        kind: 'remix',
        models: [model],
        owner,
        projectId: 'project-a',
      })
    ).resolves.toBe(false);

    expect(galleryApi.galleryImages.metadata).not.toHaveBeenCalled();
    expect(oldCommands.setSettings).not.toHaveBeenCalled();
    expect(oldCommands.add).not.toHaveBeenCalled();
  });
});
