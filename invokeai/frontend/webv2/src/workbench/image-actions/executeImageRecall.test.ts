import type { GalleryImage } from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type { GenerateModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient } from '@tanstack/react-query';
import { createInstance } from 'i18next';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryImages: {
    metadata: vi.fn(),
    resolveMany: vi.fn(),
  },
}));

vi.mock('@features/gallery', () => galleryApi);
// The transport under the gallery's shared metadata query.
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageMetadata: galleryApi.galleryImages.metadata,
}));

import { galleryImageMetadataOptions, galleryKeys } from '@features/gallery/queries';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';

import { executeImageRecall, getCurrentGenerateValues } from './executeImageRecall';

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

const t = ((key: string) => key) as unknown as TFunction;

const en = Object.values(
  import.meta.glob('../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;

const createRecallTranslator = async (lng: 'en' | 'es') => {
  const instance = createInstance();
  await instance.init({
    lng,
    fallbackLng: 'en',
    interpolation: { escapeValue: false },
    resources: {
      en: { translation: en },
      es: {
        translation: {
          widgets: {
            gallery: {
              itemActions: {
                recall: {
                  conceptsNotRestored: 'Conceptos no restaurados: {{concepts}}.',
                  nothingRecalled: 'No hay datos recuperables',
                  partiallyRecalled: 'Algunos datos no se recuperaron',
                  unnamedConcept: 'concepto sin nombre',
                  skippedConcept: {
                    ambiguous: '{{name}} (varios modelos coinciden)',
                    duplicate: '{{name}} (duplicado)',
                    incompatible: '{{name}} (incompatible con el modelo seleccionado)',
                    invalid: '{{name}} (metadatos ilegibles)',
                    modelUnavailable: '{{name}} (modelo de la imagen no disponible)',
                    unresolved: '{{name}} (no instalado)',
                  },
                },
              },
            },
          },
        },
      },
    },
  });
  return instance.t;
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

seedArchitectureCapabilities();

let queryClient: QueryClient;

describe('executeImageRecall', () => {
  beforeEach(() => {
    queryClient = new QueryClient();
    accountLifecycle.activate('test-account');
    galleryApi.galleryImages.metadata.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
  });

  it('recalls remix settings into Generate values', async () => {
    const { add, commands, setSettings } = createCommands();

    galleryApi.galleryImages.metadata.mockResolvedValue({ positive_prompt: 'recalled prompt' });

    await expect(
      executeImageRecall({
        queryClient,
        t,
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
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'success', title: 'widgets.gallery.itemActions.recall.imageTitles.remix' })
    );
  });

  it('recalls from metadata a menu or Preview Details already read, without reading it again', async () => {
    const { commands, setSettings } = createCommands();
    const recall = () =>
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key },
        image,
        kind: 'prompts',
        models: [model],
        projectId: 'project-1',
      });

    galleryApi.galleryImages.metadata.mockResolvedValue({ positive_prompt: 'recalled prompt' });

    await expect(recall()).resolves.toBe(true);
    await expect(recall()).resolves.toBe(true);

    expect(galleryApi.galleryImages.metadata).toHaveBeenCalledOnce();
    expect(setSettings).toHaveBeenLastCalledWith(
      expect.objectContaining({ positivePrompt: 'recalled prompt' }),
      'project-1'
    );
  });

  it('reads again when the read it joined is cancelled while the account is current', async () => {
    const { commands, setSettings } = createCommands();
    let resolveFirstRead: ((value: unknown) => void) | undefined;

    galleryApi.galleryImages.metadata
      .mockImplementationOnce(
        (_name: string, signal: AbortSignal) =>
          new Promise((resolve, reject) => {
            resolveFirstRead = resolve;
            signal.addEventListener('abort', () => reject(signal.reason));
          })
      )
      .mockResolvedValueOnce({ positive_prompt: 'recalled prompt' });

    // Another reader started the read; recall joins it. Unmounting a real observer is covered by the hook's browser test.
    void queryClient.prefetchQuery(galleryImageMetadataOptions(accountLifecycle.capture(), image.imageName));
    const recall = executeImageRecall({
      queryClient,
      t,
      commands,
      generateValues: { modelKey: model.key },
      image,
      kind: 'prompts',
      models: [model],
      projectId: 'project-1',
    });

    await vi.waitFor(() => expect(resolveFirstRead).toBeDefined());
    await queryClient.cancelQueries({
      queryKey: galleryKeys.imageMetadata(accountLifecycle.capture(), image.imageName),
    });

    await expect(recall).resolves.toBe(true);
    expect(galleryApi.galleryImages.metadata).toHaveBeenCalledTimes(2);
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'recalled prompt' }),
      'project-1'
    );
  });

  it('reports an image the server no longer has as carrying no metadata, and a failed read as an error', async () => {
    const { add, commands, setSettings } = createCommands();
    const recall = () =>
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key },
        image,
        kind: 'prompts',
        models: [model],
        projectId: 'project-1',
      });

    galleryApi.galleryImages.metadata.mockRejectedValueOnce(new ApiError('Not Found', 404));
    await expect(recall()).resolves.toBe(false);
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'info', message: 'widgets.gallery.itemActions.recall.imageMetadataUnsupported' })
    );

    queryClient.clear();
    galleryApi.galleryImages.metadata.mockRejectedValueOnce(new ApiError('Internal Server Error', 500));
    await expect(recall()).resolves.toBe(false);
    expect(commands.notifications.reportError).toHaveBeenCalledWith(
      expect.objectContaining({ area: 'image-recall', message: 'Internal Server Error' })
    );

    // The failure left no answer behind, so the next recall reads again.
    galleryApi.galleryImages.metadata.mockResolvedValueOnce({ positive_prompt: 'recalled prompt' });
    await expect(recall()).resolves.toBe(true);
    expect(galleryApi.galleryImages.metadata).toHaveBeenCalledTimes(3);
    expect(setSettings).toHaveBeenCalledOnce();
  });

  it.each([
    { label: 'restores every recorded concept', recorded: ['ink-lora'], skipNotice: false },
    { label: 'reports concepts it could not restore', recorded: ['ink-lora', 'deleted-lora'], skipNotice: true },
  ])('$label', async ({ recorded, skipNotice }) => {
    const { add, commands, setSettings } = createCommands();
    const inkLora = { base: 'sdxl', key: 'ink-lora', name: 'Ink', type: 'lora' } as ModelConfig;
    const t = await createRecallTranslator('en');

    galleryApi.galleryImages.metadata.mockResolvedValue({
      loras: recorded.map((key) => ({ model: { base: 'sdxl', key, name: key, type: 'lora' }, weight: 0.6 })),
      model: { key: model.key },
    });

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key },
        image: { ...image, imageName: `concepts-${recorded.length}.png` },
        kind: 'all',
        models: [model, inkLora],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ loras: [{ isEnabled: true, model: inkLora, weight: 0.6 }] }),
      'project-1'
    );
    expect(add).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', title: 'Recalled image metadata' }));
    expect(add).toHaveBeenCalledTimes(skipNotice ? 2 : 1);
    if (skipNotice) {
      expect(add).toHaveBeenCalledWith({
        kind: 'info',
        message: 'Concepts not restored: deleted-lora (not installed).',
        title: 'Some image data was not recalled',
      });
    }
  });

  it('reports a recorded refiner that is not installed while applying the rest', async () => {
    const { add, commands, setSettings } = createCommands();
    const t = await createRecallTranslator('en');

    galleryApi.galleryImages.metadata.mockResolvedValue({
      model: { key: model.key },
      refiner_model: { base: 'sdxl-refiner', key: 'gone-refiner', name: 'Gone Refiner' },
      refiner_start: 0.6,
    });

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key },
        image: { ...image, imageName: 'missing-refiner.png' },
        kind: 'all',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ refinerModel: null, refinerStart: 0.8 }),
      'project-1'
    );
    expect(add).toHaveBeenCalledWith({
      kind: 'info',
      message: 'Refiner not restored: Gone Refiner is not installed; the current refiner settings were kept.',
      title: 'Some image data was not recalled',
    });
  });

  it.each([
    {
      metadata: { model: { key: model.key }, refiner_model: {} },
      message: 'Refiner not restored: unreadable metadata; the current refiner settings were kept.',
    },
    {
      metadata: { generation_mode: 'sdxl_txt2img', model: { key: 'deleted-main' } },
      message: "Refiner not restored: the image's model is not available; the current refiner settings were kept.",
    },
  ])('explains retained refiner settings when none is selected', async ({ metadata, message }) => {
    const { add, commands, setSettings } = createCommands();
    const t = await createRecallTranslator('en');
    galleryApi.galleryImages.metadata.mockResolvedValue({
      ...metadata,
      positive_prompt: 'recalled',
      refiner_start: 0.6,
    });

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key, refinerModel: null, refinerStart: 0.9 },
        image,
        kind: 'all',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'recalled', refinerModel: null, refinerStart: 0.9 }),
      'project-1'
    );
    expect(add).toHaveBeenCalledWith({
      kind: 'info',
      message,
      title: 'Some image data was not recalled',
    });
  });

  it('explains why nothing was recalled when no recorded concept can be restored', async () => {
    const { add, commands, setSettings } = createCommands();
    const t = await createRecallTranslator('en');

    galleryApi.galleryImages.metadata.mockResolvedValue({
      loras: [{ model: { base: 'sdxl', key: 'deleted-lora', name: 'Deleted', type: 'lora' }, weight: 1 }],
    });

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key },
        image: { ...image, imageName: 'unrestorable-concepts.png' },
        kind: 'remix',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledWith({
      kind: 'info',
      message: 'Concepts not restored: Deleted (not installed).',
      title: 'No recallable image data',
    });
  });

  it('localizes partial recall reasons and unnamed concepts while preserving model names', async () => {
    const { add, commands, setSettings } = createCommands();
    const inkLora = { base: 'sdxl', key: 'ink-lora', name: 'Ink & <Wash>', type: 'lora' } as ModelConfig;
    const sd1Lora = { base: 'sd-1', key: 'sd1-lora', name: 'SD1 Ink', type: 'lora' } as ModelConfig;
    const t = await createRecallTranslator('es');

    galleryApi.galleryImages.metadata.mockResolvedValue({
      model: { key: model.key },
      loras: [
        { model: { key: inkLora.key }, weight: 0.6 },
        { model: { key: inkLora.key }, weight: 1 },
        { model: { key: 'missing', name: 'Lost & <Found>' }, weight: 1 },
        { model: { key: sd1Lora.key }, weight: 1 },
        { model: { hash: 'ambiguous-hash', name: 'Grain' }, weight: 1 },
        null,
      ],
    });

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key },
        image,
        kind: 'all',
        models: [
          model,
          inkLora,
          sd1Lora,
          { ...inkLora, key: 'grain-1', hash: 'ambiguous-hash' },
          { ...inkLora, key: 'grain-2', hash: 'ambiguous-hash' },
        ],
      })
    ).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ loras: [{ isEnabled: true, model: inkLora, weight: 0.6 }] }),
      undefined
    );
    expect(add).toHaveBeenCalledTimes(2);
    expect(add).toHaveBeenLastCalledWith({
      kind: 'info',
      message:
        'Conceptos no restaurados: Ink & <Wash> (duplicado); Lost & <Found> (no instalado); SD1 Ink (incompatible con el modelo seleccionado); Grain (varios modelos coinciden); concepto sin nombre (metadatos ilegibles).',
      title: 'Algunos datos no se recuperaron',
    });
  });

  it('localizes the notice when the recorded model is unavailable and nothing can be recalled', async () => {
    const { add, commands, setSettings } = createCommands();
    const t = await createRecallTranslator('es');

    galleryApi.galleryImages.metadata.mockResolvedValue({
      model: { key: 'missing-model' },
      loras: [{ model: { key: 'missing-concept' }, weight: 1 }],
    });

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        generateValues: { modelKey: model.key },
        image,
        kind: 'remix',
        models: [model],
      })
    ).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      message: 'Conceptos no restaurados: concepto sin nombre (modelo de la imagen no disponible).',
      title: 'No hay datos recuperables',
    });
  });

  it('persists no remix until model capabilities load, and says why', async () => {
    const { add, commands, setSettings } = createCommands();
    // Initialised while the table is present, so only the recall itself is gated.
    const generateValues = getCurrentGenerateValues({
      generateValues: { modelKey: model.key },
      supportedModels: [model as GenerateModelConfig],
    });

    resetArchitectureCapabilities();
    try {
      await expect(
        executeImageRecall({
          queryClient,
          t,
          commands,
          generateValues: generateValues as unknown as Record<string, unknown>,
          image,
          kind: 'remix',
          models: [model],
          projectId: 'project-1',
        })
      ).resolves.toBe(false);
    } finally {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    }

    expect(galleryApi.galleryImages.metadata).not.toHaveBeenCalled();
    expect(setSettings).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledWith({
      kind: 'info',
      message: 'widgets.generate.capabilitiesUnavailableForRecall',
      title: 'widgets.gallery.itemActions.recall.cannotRecallImage',
    });
  });

  it('does not initialise a fresh project from fallback defaults for a prompt recall during an outage', async () => {
    const { add, commands, setSettings } = createCommands();

    galleryApi.galleryImages.metadata.mockResolvedValue({ positive_prompt: 'recalled prompt' });

    resetArchitectureCapabilities();
    try {
      await expect(
        executeImageRecall({
          queryClient,
          t,
          commands,
          // Never initialised: nothing but a model key, so the values would be synthesised.
          generateValues: { modelKey: model.key },
          image,
          kind: 'prompts',
          models: [model],
          projectId: 'project-1',
        })
      ).resolves.toBe(false);
    } finally {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    }

    // Nothing could be applied, so nothing is read: a failing read must not replace the notice with an error.
    expect(galleryApi.galleryImages.metadata).not.toHaveBeenCalled();
    expect(galleryApi.galleryImages.resolveMany).not.toHaveBeenCalled();
    expect(setSettings).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledWith({
      kind: 'info',
      message: 'widgets.generate.capabilitiesUnavailableForSetup',
      title: 'widgets.gallery.itemActions.recall.cannotRecallImage',
    });
  });

  it('uses the freshest Generate values when recalling image dimensions', async () => {
    const { commands, setSettings } = createCommands();

    await expect(
      executeImageRecall({
        queryClient,
        t,
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
        queryClient,
        t,
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

  it('keeps the current references when every recorded reference was deleted', async () => {
    const { commands, setSettings } = createCommands();
    const current = {
      config: {
        image: { original: { image: { height: 64, image_name: 'current.png', width: 64 } } },
        model: null,
        type: 'ip_adapter',
      },
      id: 'current',
      isEnabled: true,
    };

    galleryApi.galleryImages.metadata.mockResolvedValue({
      positive_prompt: 'recalled prompt',
      ref_images: [
        {
          config: { image: { height: 64, image_name: 'deleted.png', width: 64 }, type: 'qwen_image_reference_image' },
          id: 'deleted',
          isEnabled: true,
        },
      ],
    });
    galleryApi.galleryImages.resolveMany.mockResolvedValue([]);

    await executeImageRecall({
      queryClient,
      t,
      commands,
      generateValues: { modelKey: model.key, referenceImages: [current] },
      image,
      kind: 'all',
      models: [model],
      projectId: 'project-1',
    });

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        positivePrompt: 'recalled prompt',
        referenceImages: [expect.objectContaining({ id: 'current' })],
      }),
      'project-1'
    );
  });

  it('applies a prompts recall on top of an edit made while the metadata loaded', async () => {
    const { commands, setSettings } = createCommands();
    let finishRead: ((value: unknown) => void) | undefined;
    let panel: Record<string, unknown> = { modelKey: model.key, steps: 20 };
    const getGenerateValues = vi.fn(() => panel);

    galleryApi.galleryImages.metadata.mockReturnValueOnce(
      new Promise((resolve) => {
        finishRead = resolve;
      })
    );
    const recall = executeImageRecall({
      queryClient,
      t,
      commands,
      generateValues: panel,
      getGenerateValues,
      image,
      kind: 'prompts',
      models: [model],
      projectId: 'project-1',
    });

    await vi.waitFor(() => expect(finishRead).toBeDefined());
    panel = { ...panel, steps: 42 };
    finishRead!({ positive_prompt: 'recalled prompt' });

    await expect(recall).resolves.toBe(true);
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'recalled prompt', steps: 42 }),
      'project-1'
    );
  });

  it('applies on top of an edit made while the reference check was pending', async () => {
    const { commands, setSettings } = createCommands();
    let finishCheck: ((value: unknown) => void) | undefined;
    let panel: Record<string, unknown> = { modelKey: model.key, steps: 20 };
    const getGenerateValues = vi.fn(() => panel);

    galleryApi.galleryImages.metadata.mockResolvedValue({
      positive_prompt: 'recalled prompt',
      ref_images: [
        {
          config: { image: { height: 64, image_name: 'kept.png', width: 64 }, type: 'qwen_image_reference_image' },
          id: 'kept',
          isEnabled: true,
        },
      ],
    });
    galleryApi.galleryImages.resolveMany.mockReturnValueOnce(
      new Promise((resolve) => {
        finishCheck = resolve;
      })
    );
    const recall = executeImageRecall({
      queryClient,
      t,
      commands,
      generateValues: panel,
      getGenerateValues,
      image,
      kind: 'all',
      models: [model],
      projectId: 'project-1',
    });

    await vi.waitFor(() => expect(finishCheck).toBeDefined());
    panel = { ...panel, steps: 42 };
    finishCheck!([{ ...image, imageName: 'kept.png' }]);

    await expect(recall).resolves.toBe(true);
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ referenceImages: [expect.objectContaining({ id: 'kept' })], steps: 42 }),
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
      queryClient,
      t,
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
        queryClient,
        t,
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
        queryClient,
        t,
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
