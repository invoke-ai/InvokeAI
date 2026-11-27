import type { GalleryImage } from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type { GenerateModelConfig, GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient } from '@tanstack/react-query';
import { createInstance } from 'i18next';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  GALLERY_REQUEST_MAX_NAMES: 1000,
  galleryImages: {
    metadata: vi.fn(),
    resolveMany: vi.fn(),
    workflow: vi.fn(),
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
} from '@features/models/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities.testing';

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
    galleryApi.galleryImages.workflow.mockReset();
  });

  it('recalls remix settings into Generate values', async () => {
    const { add, commands, setSettings } = createCommands();

    galleryApi.galleryImages.metadata.mockResolvedValue({ positive_prompt: 'recalled prompt' });

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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

    // Another reader started the read and recall joins it; the hook's browser test covers unmounting an observer.
    void queryClient.prefetchQuery(galleryImageMetadataOptions(accountLifecycle.capture(), image.imageName));
    const recall = executeImageRecall({
      queryClient,
      t,
      commands,
      getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key, refinerModel: null, refinerStart: 0.9 }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
      models: [model as GenerateModelConfig],
      supportedModels: [model as GenerateModelConfig],
    });

    resetArchitectureCapabilities();
    try {
      await expect(
        executeImageRecall({
          queryClient,
          t,
          commands,
          getGenerateValues: () => generateValues as unknown as Record<string, unknown>,
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
          getGenerateValues: () => ({ modelKey: model.key }),
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

  it('keeps the current Generate values when recalling image dimensions', async () => {
    const { commands, setSettings } = createCommands();

    await expect(
      executeImageRecall({
        queryClient,
        t,
        commands,
        getGenerateValues: () => ({ modelKey: model.key, positivePrompt: 'current prompt' }),
        image,
        kind: 'dimensions',
        models: [model],
        projectId: 'project-1',
      })
    ).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'current prompt', height: image.height, width: image.width }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
      getGenerateValues: () => ({ modelKey: model.key, referenceImages: [current] }),
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

  it.each(['prompts', 'positivePrompt', 'negativePrompt'] as const)(
    'applies %s recall without reverting unrelated edits made while metadata loaded',
    async (kind) => {
      const { commands, setSettings } = createCommands();
      let finishRead: ((value: unknown) => void) | undefined;
      let panel: Record<string, unknown> = {
        modelKey: model.key,
        steps: 20,
        positivePrompt: 'original positive',
        negativePrompt: 'original negative',
      };
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
        getGenerateValues,
        image,
        kind,
        models: [model],
        projectId: 'project-1',
      });

      await vi.waitFor(() => expect(finishRead).toBeDefined());
      panel = {
        ...panel,
        steps: 42,
        positivePrompt: 'edited positive',
        negativePrompt: 'edited negative',
        negativePromptEnabled: false,
        promptTemplate: {
          id: 'new-template',
          name: 'New template',
          positivePrompt: '{prompt}, cinematic',
          negativePrompt: '{prompt}, lowres',
        },
        promptTemplateViewMode: true,
      };
      finishRead!({ positive_prompt: 'recalled positive', negative_prompt: 'recalled negative' });

      await expect(recall).resolves.toBe(true);
      expect(setSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          positivePrompt: kind === 'negativePrompt' ? 'edited positive, cinematic' : 'recalled positive',
          negativePrompt: kind === 'positivePrompt' ? 'edited negative, lowres' : 'recalled negative',
          negativePromptEnabled: kind !== 'positivePrompt',
          promptTemplate: null,
          promptTemplateViewMode: false,
          steps: 42,
        }),
        'project-1'
      );
    }
  );

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
      getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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
        getGenerateValues: () => ({ modelKey: model.key }),
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

  describe('a record that names no VAE, beside a panel that holds one', () => {
    const vae = { base: 'sdxl', format: 'checkpoint', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' } as ModelConfig;
    // What webv2's Generate graph writes for an SDXL run that used the main model's own VAE.
    const record = {
      generation_mode: 'sdxl_txt2img',
      model: { base: 'sdxl', hash: 'hash', key: model.key, name: model.name, type: 'main' },
      positive_prompt: 'p',
      rand_device: 'cpu',
    };
    const recallVae = async () => {
      const { commands, setSettings } = createCommands();

      galleryApi.galleryImages.metadata.mockResolvedValue(record);
      await executeImageRecall({
        queryClient,
        t,
        commands,
        getGenerateValues: () => ({ modelKey: model.key, vae }),
        image,
        kind: 'remix',
        models: [model, vae],
        projectId: 'project-1',
      });

      return (setSettings.mock.calls[0]![0] as GenerateWidgetValues).vae;
    };

    it('clears the panel VAE for a graph run, which names every VAE it loaded', async () => {
      galleryApi.galleryImages.workflow.mockResolvedValue({ graph: '{}', workflow: null });

      await expect(recallVae()).resolves.toBeNull();
    });

    it('keeps the panel VAE for a run that embeds a workflow, whose loader may have used it unrecorded', async () => {
      galleryApi.galleryImages.workflow.mockResolvedValue({ graph: '{}', workflow: '{"nodes":[]}' });

      expect((await recallVae())?.key).toBe(vae.key);
    });

    it('reads the embedded workflow once per image, whatever the number of recalls', async () => {
      galleryApi.galleryImages.workflow.mockResolvedValue({ graph: '{}', workflow: null });

      await recallVae();
      await recallVae();

      expect(galleryApi.galleryImages.workflow).toHaveBeenCalledTimes(1);
    });

    it('reads no workflow for a record that cannot vouch for its components', async () => {
      queryClient.clear();
      const { commands } = createCommands();

      galleryApi.galleryImages.metadata.mockResolvedValue({ positive_prompt: 'p' });
      await executeImageRecall({
        queryClient,
        t,
        commands,
        getGenerateValues: () => ({ modelKey: model.key }),
        image,
        kind: 'remix',
        models: [model],
        projectId: 'project-1',
      });

      expect(galleryApi.galleryImages.workflow).not.toHaveBeenCalled();
    });

    it('keeps the panel VAE when the workflow cannot be read', async () => {
      galleryApi.galleryImages.workflow.mockRejectedValue(new ApiError('Internal Server Error', 500));

      expect((await recallVae())?.key).toBe(vae.key);
    });
  });

  describe('in a fresh project whose model stores default components', () => {
    const q21 = {
      base: 'qwen-image-2-1',
      default_settings: { components: { qwen3_vl_encoder: 'q21-encoder' } },
      format: 'diffusers',
      key: 'q21-main',
      name: 'Qwen-Image-2.1',
      type: 'main',
      variant: 'qwen_image_21_base',
    } as unknown as ModelConfig;
    const encoder = {
      base: 'any',
      format: 'gguf_quantized',
      key: 'q21-encoder',
      name: 'External encoder',
      type: 'qwen3_vl_encoder',
      variant: 'qwen3_vl_8b',
    } as unknown as ModelConfig;
    const recallValues = async (
      kind: 'prompts' | 'remix',
      metadata: Record<string, unknown>,
      main: ModelConfig = q21,
      installed: ModelConfig[] = [q21, encoder]
    ) => {
      // The metadata is cached per image: each recall reads its own.
      queryClient.clear();
      const { commands, setSettings } = createCommands();

      galleryApi.galleryImages.metadata.mockResolvedValue(metadata);
      await executeImageRecall({
        queryClient,
        t,
        commands,
        // Never initialised: nothing but a model key, so the values are synthesised.
        getGenerateValues: () => ({ modelKey: main.key }),
        image,
        kind,
        models: installed,
        projectId: 'project-1',
      });

      return setSettings.mock.calls[0]![0] as GenerateWidgetValues;
    };
    const recallInto = async (kind: 'prompts' | 'remix', metadata: Record<string, unknown>) =>
      (await recallValues(kind, metadata)).qwen3VLEncoderModel;

    it('lets a remix of an image made with the bundled encoder keep the bundled one', async () => {
      await expect(
        recallInto('remix', { generation_mode: 'txt2img', model: { key: q21.key }, positive_prompt: 'bundled' })
      ).resolves.toBeNull();
    });

    it('keeps a component the image records as empty cleared, though the image names no model', async () => {
      await expect(recallInto('remix', { positive_prompt: 'no model', qwen3_vl_encoder: null })).resolves.toBeNull();
    });

    it('keeps a component source recorded as empty under its alias cleared', async () => {
      const source = {
        base: 'wan',
        format: 'diffusers',
        key: 'wan-source',
        name: 'Wan pipeline',
        type: 'main',
        variant: 't2v_a14b',
      } as unknown as ModelConfig;
      const wan = {
        base: 'wan',
        default_settings: { components: { component_source: source.key } },
        format: 'gguf_quantized',
        key: 'wan-main',
        name: 'Wan',
        type: 'main',
        variant: 't2v_a14b',
      } as unknown as ModelConfig;
      const installed = [wan, source];

      const filled = await recallValues('prompts', { positive_prompt: 'p' }, wan, installed);
      expect(filled.componentSourceModel?.key).toBe(source.key);

      const cleared = await recallValues('remix', { positive_prompt: 'p', wan_component_source: null }, wan, installed);
      expect(cleared.componentSourceModel).toBeNull();
    });

    it('fills the default for a slot whose recorded pick recall rejected', async () => {
      const qwen = (key: string, variant: string) =>
        ({
          base: 'any',
          format: 'checkpoint',
          key,
          name: key,
          type: 'qwen3_encoder',
          variant,
        }) as unknown as ModelConfig;
      const small = qwen('enc-06b', 'qwen3_06b');
      const fitting = qwen('enc-4b', 'qwen3_4b');
      const zImage = {
        base: 'z-image',
        default_settings: { components: { qwen3_encoder: fitting.key } },
        format: 'checkpoint',
        key: 'z-main',
        name: 'Z-Image',
        type: 'main',
      } as unknown as ModelConfig;

      const values = await recallValues('remix', { positive_prompt: 'p', qwen3_encoder: { key: small.key } }, zImage, [
        zImage,
        small,
        fitting,
      ]);

      expect(values.qwen3EncoderModel?.key).toBe(fitting.key);
    });

    it('lets a working default take the slot of a recorded pick whose files are missing', async () => {
      const qwen = (key: string, extra: Record<string, unknown> = {}) =>
        ({
          base: 'any',
          format: 'checkpoint',
          key,
          name: key,
          type: 'qwen3_encoder',
          variant: 'qwen3_4b',
          ...extra,
        }) as unknown as ModelConfig;
      const gone = qwen('enc-gone', { is_missing: true });
      const working = qwen('enc-working');
      const zImage = {
        base: 'z-image',
        default_settings: { components: { qwen3_encoder: working.key } },
        format: 'checkpoint',
        key: 'z-main',
        name: 'Z-Image',
        type: 'main',
      } as unknown as ModelConfig;

      const values = await recallValues('remix', { positive_prompt: 'p', qwen3_encoder: { key: gone.key } }, zImage, [
        zImage,
        gone,
        working,
      ]);

      expect(values.qwen3EncoderModel?.key).toBe(working.key);
    });

    it('leaves the default vision tower out when the image clears the encoder it hangs on', async () => {
      const tower = { ...encoder, key: 'q21-tower', name: 'Tower', type: 'qwen3_vl_vision_tower' } as ModelConfig;
      const withTower = {
        ...q21,
        default_settings: { components: { qwen3_vl_encoder: encoder.key, qwen3_vl_vision_tower: tower.key } },
      } as unknown as ModelConfig;
      const installed = [withTower, encoder, tower];

      const filled = await recallValues('remix', { positive_prompt: 'p' }, withTower, installed);
      expect(filled.qwen3VLVisionTowerModel?.key).toBe(tower.key);

      const cleared = await recallValues(
        'remix',
        { positive_prompt: 'p', qwen3_vl_encoder: null },
        withTower,
        installed
      );
      expect(cleared.qwen3VLEncoderModel).toBeNull();
      expect(cleared.qwen3VLVisionTowerModel).toBeNull();
    });

    it('lets a working default VAE take the place of a recorded VAE whose files are missing', async () => {
      const vae = (key: string, extra: Record<string, unknown> = {}) =>
        ({ base: 'flux', format: 'checkpoint', key, name: key, type: 'vae', ...extra }) as unknown as ModelConfig;
      const gone = vae('flux-vae-gone', { is_missing: true });
      const working = vae('flux-vae-working');
      const flux = {
        base: 'flux',
        default_settings: { vae: working.key },
        format: 'checkpoint',
        key: 'flux-main',
        name: 'FLUX',
        type: 'main',
      } as unknown as ModelConfig;

      const values = await recallValues('remix', { positive_prompt: 'p', vae: { key: gone.key } }, flux, [
        flux,
        gone,
        working,
      ]);

      expect(values.vae?.key).toBe(working.key);
    });

    it('starts a fresh project on a selectable main model, not on a standalone branch the metadata could name', async () => {
      const make = (key: string, base: string, extra: Record<string, unknown> = {}) =>
        ({ base, format: 'checkpoint', key, name: key, type: 'main', ...extra }) as unknown as ModelConfig;
      const gone = make('gone', 'sdxl', { is_missing: true });
      const branch = make('branch', 'ideogram-4', { branch: 'unconditional' });
      const sdxl = make('sdxl-ok', 'sdxl');

      const values = await recallValues('remix', { positive_prompt: 'p' }, make('unset', 'sdxl'), [gone, branch, sdxl]);

      expect(values.model.key).toBe('sdxl-ok');
      // The two families default to different step counts, so the settings show which model they were built for.
      expect(getDefaultGenerateSettings(branch as never).steps).not.toBe(
        getDefaultGenerateSettings(sdxl as never).steps
      );
      expect(values.steps).toBe(getDefaultGenerateSettings(sdxl as never).steps);
    });

    it('keeps a VAE the image records as empty cleared', async () => {
      const vae = { base: 'flux', format: 'checkpoint', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' } as ModelConfig;
      const flux = {
        base: 'flux',
        default_settings: { vae: vae.key },
        format: 'checkpoint',
        key: 'flux-main',
        name: 'FLUX',
        type: 'main',
      } as unknown as ModelConfig;

      expect((await recallValues('prompts', { positive_prompt: 'p' }, flux, [flux, vae])).vae?.key).toBe(vae.key);
      expect((await recallValues('remix', { positive_prompt: 'p', vae: null }, flux, [flux, vae])).vae).toBeNull();
    });

    it('still fills the saved default when the recall does not touch the model', async () => {
      await expect(recallInto('prompts', { positive_prompt: 'prompt only' })).resolves.toMatchObject({
        key: encoder.key,
      });
    });
  });
});

const legacyIPAdapter = () => ({
  ip_adapter_model: { model_name: 'Portrait', base_model: 'sdxl', model_type: 'ip_adapter' },
  image: { image_name: 'portrait.png' },
  clip_vision_model: 'ViT-G',
  method: 'style_precise',
  weight: 0.65,
  begin_step_percent: 0.2,
  end_step_percent: 0.9,
});
const portraitModel = { key: 'portrait-installed', name: 'Portrait', base: 'sdxl', type: 'ip_adapter' } as ModelConfig;

it.each(['all', 'remix'] as const)(
  'hydrates v4 name-only IP-Adapters for Gallery %s without losing adapter settings',
  async (kind) => {
    accountLifecycle.activate('ip-recall-tests');
    const { commands, setSettings, add } = createCommands();
    const client = new QueryClient();
    galleryApi.galleryImages.metadata.mockResolvedValue({ ipAdapters: [legacyIPAdapter()] });
    galleryApi.galleryImages.resolveMany
      .mockReset()
      .mockResolvedValue([{ ...image, imageName: 'portrait.png', width: 333, height: 222 }]);
    expect(
      await executeImageRecall({
        commands,
        getGenerateValues: () => ({ modelKey: model.key }),
        image,
        kind,
        models: [model, portraitModel],
        queryClient: client,
        t,
      })
    ).toBe(true);
    expect(galleryApi.galleryImages.resolveMany).toHaveBeenCalledExactlyOnceWith(
      ['portrait.png'],
      expect.any(AbortSignal)
    );
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        referenceImages: [
          {
            id: 'legacy-ip-adapter-0',
            isEnabled: true,
            config: {
              type: 'ip_adapter',
              model: portraitModel,
              image: { original: { image: { image_name: 'portrait.png', width: 333, height: 222 } } },
              weight: 0.65,
              clipVisionModel: 'ViT-G',
              method: 'style_precise',
              beginEndStepPct: [0.2, 0.9],
            },
          },
        ],
      }),
      undefined
    );
    expect(add).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: 'widgets.gallery.itemActions.recall.partiallyRecalled' })
    );
  }
);

it.each(['deleted', 'lookup-failed', 'unsupported-weight', 'multi-image', 'missing-model', 'malformed'] as const)(
  'keeps existing references and recalls independent Generate fields when v4 adapters are %s',
  async (failure) => {
    accountLifecycle.activate('ip-failure-tests');
    const { commands, setSettings, add } = createCommands();
    const current = {
      id: 'current',
      isEnabled: true,
      config: {
        type: 'ip_adapter',
        model: portraitModel,
        image: { original: { image: { image_name: 'current.png', width: 100, height: 100 } } },
        weight: 0.7,
        beginEndStepPct: [0, 1],
        method: 'full',
        clipVisionModel: 'ViT-H',
      },
    };
    const adapter = legacyIPAdapter();
    const bad =
      failure === 'unsupported-weight'
        ? { ...adapter, weight: [0.2, 0.9] }
        : failure === 'multi-image'
          ? { ...adapter, image: [{ image_name: 'one.png' }, { image_name: 'two.png' }] }
          : failure === 'missing-model'
            ? { ...adapter, ip_adapter_model: { key: 'deleted' } }
            : failure === 'malformed'
              ? { ...adapter, end_step_percent: -1 }
              : adapter;
    galleryApi.galleryImages.metadata.mockResolvedValue({ positive_prompt: 'independent', ipAdapters: [bad] });
    galleryApi.galleryImages.resolveMany.mockReset();
    if (failure === 'lookup-failed') {
      galleryApi.galleryImages.resolveMany.mockRejectedValue(new ApiError('Unavailable', 503));
    } else {
      galleryApi.galleryImages.resolveMany.mockResolvedValue(
        failure === 'deleted' ? [] : [{ ...image, imageName: 'portrait.png' }]
      );
    }
    expect(
      await executeImageRecall({
        commands,
        getGenerateValues: () => ({ modelKey: model.key, referenceImages: [current] }),
        image,
        kind: 'all',
        models: [model, portraitModel],
        queryClient: new QueryClient(),
        t,
      })
    ).toBe(true);
    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        positivePrompt: 'independent',
        referenceImages: [expect.objectContaining({ id: 'current' })],
      }),
      undefined
    );
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'widgets.gallery.itemActions.recall.partiallyRecalled' })
    );
  }
);

it('keeps supported IP entries while reporting malformed and unsupported siblings', async () => {
  accountLifecycle.activate('ip-partial-tests');
  const { commands, setSettings, add } = createCommands();
  galleryApi.galleryImages.metadata.mockResolvedValue({
    ipAdapters: [legacyIPAdapter(), { ...legacyIPAdapter(), weight: [1, 2] }, null],
  });
  galleryApi.galleryImages.resolveMany.mockReset().mockResolvedValue([{ ...image, imageName: 'portrait.png' }]);
  expect(
    await executeImageRecall({
      commands,
      getGenerateValues: () => ({ modelKey: model.key }),
      image,
      kind: 'remix',
      models: [model, portraitModel],
      queryClient: new QueryClient(),
      t,
    })
  ).toBe(true);
  expect(setSettings.mock.calls[0]![0].referenceImages).toHaveLength(1);
  expect(add).toHaveBeenCalledWith(
    expect.objectContaining({ title: 'widgets.gallery.itemActions.recall.partiallyRecalled' })
  );
});

it('uses sequential bounded reference batches and lets Generate recall survive a second-batch failure', async () => {
  accountLifecycle.activate('ip-batch-tests');
  const { commands, setSettings, add } = createCommands();
  galleryApi.galleryImages.metadata.mockResolvedValue({
    positive_prompt: 'independent',
    ipAdapters: Array.from({ length: 1002 }, (_, index) => ({
      ...legacyIPAdapter(),
      image: { image_name: `ref-${index % 1001}.png` },
    })),
  });
  galleryApi.galleryImages.resolveMany
    .mockReset()
    .mockImplementationOnce((names: string[]) => Promise.resolve(names.map((imageName) => ({ ...image, imageName }))))
    .mockRejectedValueOnce(new TypeError('Failed to fetch'));
  expect(
    await executeImageRecall({
      commands,
      getGenerateValues: () => ({ modelKey: model.key }),
      image,
      kind: 'all',
      models: [model, portraitModel],
      queryClient: new QueryClient(),
      t,
    })
  ).toBe(true);
  expect(galleryApi.galleryImages.resolveMany.mock.calls.map(([names]) => names.length)).toEqual([1000, 1]);
  expect(setSettings).toHaveBeenCalledWith(
    expect.objectContaining({ positivePrompt: 'independent', referenceImages: [] }),
    undefined
  );
  expect(add).toHaveBeenCalledWith(
    expect.objectContaining({ title: 'widgets.gallery.itemActions.recall.partiallyRecalled' })
  );
});

it('abandons name-only reference hydration after account expiry before another batch or Generate commit', async () => {
  accountLifecycle.activate('ip-expiry-tests');
  const { commands, setSettings, add } = createCommands();
  galleryApi.galleryImages.metadata.mockResolvedValue({
    positive_prompt: 'wrong account',
    ipAdapters: Array.from({ length: 1001 }, (_, index) => ({
      ...legacyIPAdapter(),
      image: { image_name: `ref-${index}.png` },
    })),
  });
  galleryApi.galleryImages.resolveMany.mockReset().mockImplementation((names: string[]) => {
    accountLifecycle.activate('different-account');
    return Promise.resolve(names.map((imageName) => ({ ...image, imageName })));
  });
  expect(
    await executeImageRecall({
      commands,
      getGenerateValues: () => ({ modelKey: model.key }),
      image,
      kind: 'all',
      models: [model, portraitModel],
      queryClient: new QueryClient(),
      t,
    })
  ).toBe(false);
  expect(galleryApi.galleryImages.resolveMany).toHaveBeenCalledOnce();
  expect(setSettings).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('surfaces reference asset mapping errors instead of treating them as recoverable transport failures', async () => {
  accountLifecycle.activate('ip-mapping-tests');
  const { commands, setSettings, add } = createCommands();
  galleryApi.galleryImages.metadata.mockResolvedValue({
    positive_prompt: 'must not hide the error',
    ipAdapters: [legacyIPAdapter()],
  });
  galleryApi.galleryImages.resolveMany.mockReset().mockRejectedValue(new TypeError('Invalid gallery image DTO'));
  expect(
    await executeImageRecall({
      commands,
      getGenerateValues: () => ({ modelKey: model.key }),
      image,
      kind: 'all',
      models: [model, portraitModel],
      queryClient: new QueryClient(),
      t,
    })
  ).toBe(false);
  expect(setSettings).not.toHaveBeenCalled();
  expect(commands.notifications.reportError).toHaveBeenCalledWith(
    expect.objectContaining({ message: 'Invalid gallery image DTO' })
  );
  expect(add).not.toHaveBeenCalled();
});

it('reports bounded reference truncation while hydrating the supported records in one deduplicated read', async () => {
  accountLifecycle.activate('ip-limit-tests');
  const { commands, setSettings, add } = createCommands();
  galleryApi.galleryImages.metadata.mockResolvedValue({
    ipAdapters: Array.from({ length: 10001 }, () => legacyIPAdapter()),
  });
  galleryApi.galleryImages.resolveMany.mockReset().mockResolvedValue([{ ...image, imageName: 'portrait.png' }]);
  expect(
    await executeImageRecall({
      commands,
      getGenerateValues: () => ({ modelKey: model.key }),
      image,
      kind: 'all',
      models: [model, portraitModel],
      queryClient: new QueryClient(),
      t,
    })
  ).toBe(true);
  expect(galleryApi.galleryImages.resolveMany).toHaveBeenCalledExactlyOnceWith(
    ['portrait.png'],
    expect.any(AbortSignal)
  );
  const references = setSettings.mock.calls[0]![0].referenceImages;
  expect(references).toHaveLength(10000);
  expect(references.at(-1).id).toBe('legacy-ip-adapter-9999');
  expect(add).toHaveBeenCalledWith(
    expect.objectContaining({ title: 'widgets.gallery.itemActions.recall.partiallyRecalled' })
  );
});
