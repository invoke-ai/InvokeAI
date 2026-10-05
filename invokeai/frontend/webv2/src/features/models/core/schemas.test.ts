import { describe, expect, it } from 'vitest';

import { isAbsoluteModelPath, mainDefaultSettingsSchema, modelPathSchema, resolveModelAbsolutePath } from './schemas';

describe('isAbsoluteModelPath', () => {
  it.each([
    '/models/sdxl.safetensors',
    'C:\\models\\sdxl.safetensors',
    'C:/models/sdxl.safetensors',
    '\\\\nas\\models\\x.ckpt',
  ])('accepts the absolute path %s', (path) => {
    expect(isAbsoluteModelPath(path)).toBe(true);
  });

  it.each(['models/sdxl.safetensors', './models/sdxl.safetensors', '../x.ckpt', 'sdxl.safetensors', ''])(
    'rejects the relative path %j',
    (path) => {
      expect(isAbsoluteModelPath(path)).toBe(false);
    }
  );
});

describe('modelPathSchema', () => {
  it('trims and accepts an absolute path', () => {
    expect(modelPathSchema.parse('  /models/x.safetensors  ')).toBe('/models/x.safetensors');
  });

  it('rejects empty and relative paths', () => {
    expect(modelPathSchema.safeParse('   ').success).toBe(false);
    expect(modelPathSchema.safeParse('models/x.safetensors').success).toBe(false);
  });
});

describe('resolveModelAbsolutePath', () => {
  it('resolves managed relative paths against the models directory', () => {
    expect(resolveModelAbsolutePath('sdxl/main/model.safetensors', '/data/models')).toBe(
      '/data/models/sdxl/main/model.safetensors'
    );
    expect(resolveModelAbsolutePath('sdxl/main/model.safetensors', '/data/models/')).toBe(
      '/data/models/sdxl/main/model.safetensors'
    );
  });

  it('keeps absolute in-place paths and falls back when the directory is unknown', () => {
    expect(resolveModelAbsolutePath('/home/user/model.safetensors', '/data/models')).toBe(
      '/home/user/model.safetensors'
    );
    expect(resolveModelAbsolutePath('sdxl/model.safetensors', null)).toBe('sdxl/model.safetensors');
  });
});

describe('mainDefaultSettingsSchema', () => {
  it('accepts the guidance this app itself stores for FLUX.1 Fill', () => {
    // FLUX Fill defaults to guidance 30; editing unrelated fields must accept that stored value.
    const result = mainDefaultSettingsSchema.safeParse({
      cfgRescaleMultiplier: null,
      cfgScale: 1,
      guidance: 30,
      height: 1024,
      scheduler: 'euler',
      steps: 50,
      vae: null,
      vaePrecision: null,
      width: 1024,
    });

    expect(result.success).toBe(true);
  });

  it('still rejects a guidance below what the record allows', () => {
    // `MainModelDefaultSettings.guidance` is `ge=1`; the floor is the record's, not invented.
    const result = mainDefaultSettingsSchema.safeParse({
      cfgRescaleMultiplier: null,
      cfgScale: 1,
      guidance: 0,
      height: 1024,
      scheduler: null,
      steps: 50,
      vae: null,
      vaePrecision: null,
      width: 1024,
    });

    expect(result.success).toBe(false);
  });
});
