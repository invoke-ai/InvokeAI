import { describe, expect, it } from 'vitest';

import { isAbsoluteModelPath, modelPathSchema } from './schemas';

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
