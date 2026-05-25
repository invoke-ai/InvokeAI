import { describe, expect, it } from 'vitest';

import {
  DETAILER_DINO_MODELS,
  DETAILER_SAM_MODELS,
  getDetailerDinoModelOptions,
  getDetailerSamModelOptions,
} from './detailerModelOptions';

describe('detailer model options', () => {
  const t = (key: string) => key;

  it('uses localized DINO model label keys instead of raw model ids', () => {
    const options = getDetailerDinoModelOptions(t);

    expect(options).toHaveLength(DETAILER_DINO_MODELS.length);
    for (const option of options) {
      expect(option.label).toBe(`parameters.faceDetailer.dinoModels.${option.value}`);
      expect(option.label).not.toBe(option.value);
    }
  });

  it('uses localized SAM model label keys instead of raw model ids', () => {
    const options = getDetailerSamModelOptions(t);

    expect(options).toHaveLength(DETAILER_SAM_MODELS.length);
    for (const option of options) {
      expect(option.label).toBe(`parameters.faceDetailer.samModels.${option.value}`);
      expect(option.label).not.toBe(option.value);
    }
  });
});
