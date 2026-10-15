import type { ComponentModelConfig, GenerateModelConfig } from '@features/generation/contracts';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { accountLifecycle, captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryImages: {
    metadata: vi.fn(),
    resolveMany: vi.fn(),
  },
}));

vi.mock('@features/gallery', () => galleryApi);

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';

import { getCurrentGenerateValues } from './executeImageRecall';
import { executeRecallParameters } from './executeRecallParameters';

const sdxl = { base: 'sdxl', hash: 'h', key: 'sdxl-main', name: 'SDXL', type: 'main' } as ComponentModelConfig;
const sd1 = { base: 'sd-1', hash: 'h1', key: 'sd1-main', name: 'SD 1', type: 'main' } as ComponentModelConfig;
const t = ((key: string) => key) as unknown as TFunction;

/** Runs `fn` with the capability table absent, the way a failed boot fetch leaves it. */
const withoutCapabilities = async <T>(fn: () => Promise<T>): Promise<T> => {
  resetArchitectureCapabilities();
  try {
    return await fn();
  } finally {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  }
};
const galleryImage = (imageName: string) => ({ height: 512, imageName, width: 512 });
const image = (imageName: string) => ({ height: 512, image_name: imageName, width: 512 });
const existingReference = {
  config: { image: { original: { image: image('existing.png') } }, model: null, type: 'ip_adapter' },
  id: 'existing',
  isEnabled: true,
};

const createCommands = () => {
  const add = vi.fn();
  const reportError = vi.fn();
  const setSettings = vi.fn();
  const commands = {
    generation: { setSettings } as unknown as WorkbenchCommands['generation'],
    notifications: { add, reportError } as unknown as WorkbenchCommands['notifications'],
  };

  return { add, commands, reportError, setSettings };
};

const committedReferenceNames = (setSettings: ReturnType<typeof vi.fn>) => {
  const [values] = setSettings.mock.calls[0] ?? [];
  const { referenceImages } = values as {
    referenceImages: { config: { image: { original: { image: { image_name: string } } } } }[];
  };
  return referenceImages.map((reference) => reference.config.image.original.image.image_name);
};

const run = (
  commands: ReturnType<typeof createCommands>['commands'],
  parameters: Record<string, unknown>,
  {
    generateValues = { modelKey: sdxl.key },
    getGenerateValues,
    models = [sdxl],
    owner,
  }: {
    generateValues?: Record<string, unknown>;
    getGenerateValues?: () => Record<string, unknown> | null;
    models?: ComponentModelConfig[];
    owner?: AccountScope;
  } = {}
) =>
  executeRecallParameters({
    commands,
    getGenerateValues: getGenerateValues ?? (() => generateValues),
    models,
    owner: owner ?? captureAccountScope(),
    parameters,
    projectId: 'project-1',
    t,
  });

seedArchitectureCapabilities();

describe('executeRecallParameters', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    galleryApi.galleryImages.resolveMany.mockReset();
  });

  it('commits the built values to the target project and reports what was applied', async () => {
    const { add, commands, setSettings } = createCommands();

    await expect(run(commands, { positive_prompt: 'a cat', refiner_model: 'refiner', steps: 20 })).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'a cat', steps: 20 }),
      'project-1'
    );
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'success',
        message: '2 fields applied to Generate.',
        title: 'Recalled parameters',
      })
    );
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'info',
        message: 'refiner_model: not supported by the Generate panel',
        title: 'Some recalled parameters were not applied',
      })
    );
  });

  it('explains why nothing was applied instead of committing unchanged values', async () => {
    const { add, commands, setSettings } = createCommands();

    await expect(run(commands, { denoise_strength: 0.4 })).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'info',
        message: 'denoise_strength: not supported by the Generate panel',
        title: 'No recalled parameters applied',
      })
    );
  });

  describe('while the capability table is absent', () => {
    // Initialised while the table is present, so the project already holds SDXL's real values.
    const initialised = () =>
      getCurrentGenerateValues({
        generateValues: { modelKey: sdxl.key },
        supportedModels: [sdxl as unknown as GenerateModelConfig],
      }) as unknown as Record<string, unknown> & { height: number; width: number };

    it('applies no model switch, and says why', async () => {
      const { add, commands, setSettings } = createCommands();
      const generateValues = initialised();

      await expect(
        withoutCapabilities(() => run(commands, { model: sd1.key }, { generateValues, models: [sdxl, sd1] }))
      ).resolves.toBe(false);

      expect(setSettings).not.toHaveBeenCalled();
      expect(add).toHaveBeenCalledWith({
        kind: 'info',
        message: 'widgets.generate.capabilitiesUnavailableForRecall',
        title: 'Cannot apply recalled parameters',
      });
    });

    it('withholds a dimension reset, which would restore the fallback canvas', async () => {
      const { commands, setSettings } = createCommands();
      const generateValues = initialised();

      await expect(withoutCapabilities(() => run(commands, { width: null }, { generateValues }))).resolves.toBe(false);

      expect(setSettings).not.toHaveBeenCalled();
    });

    it('still applies prompts and seed, which read no policy', async () => {
      const { commands, setSettings } = createCommands();
      const generateValues = initialised();

      await expect(
        withoutCapabilities(() => run(commands, { positive_prompt: 'a cat', seed: 7 }, { generateValues }))
      ).resolves.toBe(true);

      expect(setSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          height: generateValues.height,
          positivePrompt: 'a cat',
          seed: 7,
          width: generateValues.width,
        }),
        'project-1'
      );
    });

    it('does not initialise a fresh project from fallback defaults', async () => {
      const { add, commands, setSettings } = createCommands();

      await expect(withoutCapabilities(() => run(commands, { positive_prompt: 'a cat' }))).resolves.toBe(false);

      expect(setSettings).not.toHaveBeenCalled();
      expect(add).toHaveBeenCalledWith({
        kind: 'info',
        message: 'widgets.generate.capabilitiesUnavailableForSetup',
        title: 'Cannot apply recalled parameters',
      });
    });
  });

  it('asks for a supported model when the project has none selected', async () => {
    const { add, commands, setSettings } = createCommands();

    await expect(run(commands, { steps: 20 }, { models: [] })).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'info', title: 'Cannot apply recalled parameters' })
    );
  });

  it('drops recalled reference images whose files are no longer in the gallery and names them', async () => {
    const { add, commands, setSettings } = createCommands();
    galleryApi.galleryImages.resolveMany.mockResolvedValue([galleryImage('present.png')]);

    await expect(
      run(commands, { reference_images: [{ image: image('present.png') }, { image: image('deleted.png') }] })
    ).resolves.toBe(true);

    expect(galleryApi.galleryImages.resolveMany).toHaveBeenCalledWith(
      ['present.png', 'deleted.png'],
      expect.any(AbortSignal)
    );
    expect(committedReferenceNames(setSettings)).toEqual(['present.png']);
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'info', message: 'reference_images: not found (deleted.png)' })
    );
  });

  it('keeps the existing list when every recalled reference image is missing', async () => {
    const { add, commands, setSettings } = createCommands();
    galleryApi.galleryImages.resolveMany.mockResolvedValue([]);

    await expect(
      run(
        commands,
        { reference_images: [{ image: image('deleted.png') }] },
        {
          generateValues: { modelKey: sdxl.key, referenceImages: [existingReference] },
        }
      )
    ).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'info',
        message: 'reference_images: not found (deleted.png)',
        title: 'No recalled parameters applied',
      })
    );
  });

  it('keeps the existing list when a replacement fails entirely but other fields still apply', async () => {
    const { add, commands, setSettings } = createCommands();
    galleryApi.galleryImages.resolveMany.mockResolvedValue([]);

    await expect(
      run(
        commands,
        { positive_prompt: 'applied anyway', reference_images: [{ image: image('deleted.png') }] },
        { generateValues: { modelKey: sdxl.key, referenceImages: [existingReference] } }
      )
    ).resolves.toBe(true);

    expect(setSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'applied anyway' }),
      'project-1'
    );
    expect(committedReferenceNames(setSettings)).toEqual(['existing.png']);
    expect(add).toHaveBeenCalledWith(expect.objectContaining({ message: '1 field applied to Generate.' }));
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'info', message: 'reference_images: not found (deleted.png)' })
    );
  });

  it('verifies only the appended images, never the ones the project already had', async () => {
    const { commands, setSettings } = createCommands();
    galleryApi.galleryImages.resolveMany.mockResolvedValue([galleryImage('new.png')]);

    await expect(
      run(
        commands,
        { append: true, reference_images: [{ image: image('new.png') }] },
        {
          generateValues: { modelKey: sdxl.key, referenceImages: [existingReference] },
        }
      )
    ).resolves.toBe(true);

    expect(galleryApi.galleryImages.resolveMany).toHaveBeenCalledWith(['new.png'], expect.any(AbortSignal));
    expect(committedReferenceNames(setSettings)).toEqual(['existing.png', 'new.png']);
  });

  it('does not commit when the account changes during the gallery lookup', async () => {
    const { add, commands, setSettings } = createCommands();
    galleryApi.galleryImages.resolveMany.mockImplementation(() => {
      accountLifecycle.activate('someone-else');
      return Promise.resolve([galleryImage('present.png')]);
    });

    await expect(run(commands, { reference_images: [{ image: image('present.png') }] })).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  });

  it('does not commit when the project closes during the gallery lookup', async () => {
    const { add, commands, setSettings } = createCommands();
    let projectOpen = true;
    galleryApi.galleryImages.resolveMany.mockImplementation(() => {
      projectOpen = false;
      return Promise.resolve([galleryImage('present.png')]);
    });

    await expect(
      run(
        commands,
        { reference_images: [{ image: image('present.png') }] },
        {
          getGenerateValues: () => (projectOpen ? { modelKey: sdxl.key } : null),
        }
      )
    ).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  });

  it('reports a failed gallery lookup as an error', async () => {
    const { commands, reportError, setSettings } = createCommands();
    galleryApi.galleryImages.resolveMany.mockRejectedValue(new Error('gallery offline'));

    await expect(run(commands, { reference_images: [{ image: image('present.png') }] })).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(reportError).toHaveBeenCalledWith(
      expect.objectContaining({ area: 'recall-parameters', message: 'gallery offline', projectId: 'project-1' })
    );
  });

  it('does nothing once the account that received the event is gone', async () => {
    const { add, commands, setSettings } = createCommands();
    const owner = captureAccountScope();
    accountLifecycle.activate('someone-else');

    await expect(run(commands, { steps: 20 }, { owner })).resolves.toBe(false);

    expect(setSettings).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  });
});
