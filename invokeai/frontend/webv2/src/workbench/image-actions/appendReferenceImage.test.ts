import type { MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { getDefaultGenerateSettings } from '@features/generation/settings';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import { appendReferenceImage, type AppendReferenceImageResult } from './appendReferenceImage';

const sd1Model: MainModelConfig = {
  base: 'sd-1',
  hash: 'model-hash',
  key: 'sd1-key',
  name: 'SD1 Model',
  type: 'main',
};

const models = [sd1Model] as ModelConfig[];

const image = { height: 512, imageName: 'bbox-upload.png', width: 768 };

const generateValuesFor = (referenceImages: unknown[] = []): Record<string, unknown> => ({
  ...getDefaultGenerateSettings(sd1Model),
  model: sd1Model,
  modelKey: sd1Model.key,
  referenceImages,
});

seedArchitectureCapabilities();

describe('appendReferenceImage', () => {
  it('appends a reference image built from the flat upload result', () => {
    const result = appendReferenceImage({ generateValues: generateValuesFor(), image, models });

    if (result.status !== 'appended') {
      throw new Error(`Expected appended, got ${result.status}`);
    }
    expect(result.referenceImages).toHaveLength(1);
    expect(result.referenceImages[0]).toMatchObject({
      config: { image: { original: { image: { height: 512, image_name: 'bbox-upload.png', width: 768 } } } },
      id: expect.any(String) as string,
      isEnabled: true,
    });
  });

  it('returns full once the model reference image limit is reached', () => {
    // SD 1's served limit.
    const limit = 5;
    let referenceImages: unknown[] = [];

    for (let index = 0; index < limit; index += 1) {
      const result: AppendReferenceImageResult = appendReferenceImage({
        generateValues: generateValuesFor(referenceImages),
        image: { ...image, imageName: `image-${index}.png` },
        models,
      });
      if (result.status !== 'appended') {
        throw new Error(`Expected append ${index} to succeed`);
      }
      referenceImages = result.referenceImages as unknown[];
    }

    expect(appendReferenceImage({ generateValues: generateValuesFor(referenceImages), image, models })).toEqual({
      status: 'full',
    });
  });

  it('returns unsupported when the model cannot take reference images', () => {
    const unsupportedModel = { base: 'sd-2', hash: 'hash-2', key: 'sd2-key', name: 'SD2', type: 'main' };

    expect(
      appendReferenceImage({
        generateValues: { model: unsupportedModel, modelKey: unsupportedModel.key },
        image,
        models: [unsupportedModel] as ModelConfig[],
      })
    ).toEqual({ status: 'unsupported' });
  });

  it('falls back to the first supported model when values are incomplete', () => {
    const result = appendReferenceImage({ generateValues: {}, image, models });

    expect(result.status).toBe('appended');
  });
});
