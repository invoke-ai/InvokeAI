import { describe, expect, it } from 'vitest';

import { getGenerateNodeProvenance } from './graphProvenance';

describe('getGenerateNodeProvenance', () => {
  it('maps fixed generate node ids to settings', () => {
    expect(getGenerateNodeProvenance('denoise_latents', 'steps')).toEqual({
      labelKey: 'graphPreview.provenance.steps',
      settingKey: 'steps',
    });
    expect(getGenerateNodeProvenance('seed', 'value')?.settingKey).toBe('seed');
    expect(getGenerateNodeProvenance('positive_prompt', 'value')?.settingKey).toBe('positivePrompt');
    expect(getGenerateNodeProvenance('noise', 'width')?.labelKey).toBe('graphPreview.provenance.size');
    expect(getGenerateNodeProvenance('model_loader', 'model')?.settingKey).toBe('modelKey');
  });

  it('maps stabilized lora selector ids via prefix', () => {
    expect(getGenerateNodeProvenance('lora_selector', 'lora')?.settingKey).toBe('loras');
    expect(getGenerateNodeProvenance('lora_selector_2', 'weight')?.settingKey).toBe('loras');
  });

  it('returns null for unmapped fields', () => {
    expect(getGenerateNodeProvenance('denoise_latents', 'latents')).toBeNull();
    expect(getGenerateNodeProvenance('core_metadata', 'steps')).toBeNull();
  });
});
