import { describe, expect, it } from 'vitest';

import {
  isAbsoluteModelPath,
  canonicalLocalPath,
  isInModelsDir,
  mainDefaultSettingsSchema,
  modelPathSchema,
  resolveModelAbsolutePath,
} from './schemas';

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

describe('mainDefaultSettingsSchema default components', () => {
  const parse = (components: Record<string, string> | null) =>
    mainDefaultSettingsSchema.safeParse({
      cfgRescaleMultiplier: null,
      cfgScale: null,
      components,
      guidance: null,
      height: null,
      scheduler: null,
      steps: null,
      vae: null,
      vaePrecision: null,
      width: null,
    });

  it('accepts slot names with a model key each, or none', () => {
    expect(parse({ qwen3_5_encoder: 'key-1', t5_encoder: 'key-2' }).success).toBe(true);
    expect(parse(null).success).toBe(true);
  });

  it.each([
    ['the VAE, which has its own field', { vae: 'key' }],
    ['a slot name outside a-z, 0-9 and _', { 'T5-Encoder': 'key' }],
    ['a blank model key', { t5_encoder: '  ' }],
    ['more than sixteen entries', Object.fromEntries(Array.from({ length: 17 }, (_, index) => [`slot_${index}`, 'k']))],
  ])('rejects %s', (_label, components) => {
    expect(parse(components).success).toBe(false);
  });
});

describe('canonicalLocalPath', () => {
  it.each([
    ['C:\\Users\\Me\\x.safetensors', 'c:/users/me/x.safetensors'],
    ['  C:/Users/Me/loras/  ', 'c:/users/me/loras'],
    ['C:\\', 'c:'],
    ['\\\\NAS\\Models\\', '//nas/models'],
    ['/home/Me/Models//', '/home/Me/Models'],
    ['loras/X.safetensors', 'loras/X.safetensors'],
  ])('spells %j as %j', (path, canonical) => {
    expect(canonicalLocalPath(path)).toBe(canonical);
  });
});

describe('isInModelsDir', () => {
  it.each([
    ['/data/models', '/data/models'],
    ['/data/models/', '/data/models'],
    ['/data/models/sdxl/model.safetensors', '/data/models'],
    ['/data/models/sdxl/model.safetensors', '/data/models/'],
    ['C:\\Invoke\\Models\\sdxl.safetensors', 'C:/invoke/models'],
    ['c:/invoke/models', 'C:/Invoke/Models'],
    ['\\\\NAS\\Models\\sdxl.safetensors', '//nas/models'],
  ])('places %s in the models folder %s', (path, modelsDir) => {
    expect(isInModelsDir(path, modelsDir)).toBe(true);
  });

  it.each([
    ['/data/models-extra/model.safetensors', '/data/models'],
    ['/data/Models/model.safetensors', '/data/models'],
    ['/data', '/data/models'],
    ['/data/models/../elsewhere/model.safetensors', '/data/models'],
    ['/data/models/model.safetensors', null],
  ])('keeps %s outside the models folder %s', (path, modelsDir) => {
    expect(isInModelsDir(path, modelsDir)).toBe(false);
  });
});
