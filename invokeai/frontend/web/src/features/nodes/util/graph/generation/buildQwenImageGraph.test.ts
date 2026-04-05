import { describe, expect, it } from 'vitest';

import { isQwenImageEditModel } from './buildQwenImageGraph';

describe('isQwenImageEditModel', () => {
  it('returns true for edit variant', () => {
    expect(isQwenImageEditModel({ variant: 'edit' })).toBe(true);
  });

  it('returns false for generate variant', () => {
    expect(isQwenImageEditModel({ variant: 'generate' })).toBe(false);
  });

  it('returns false when variant is null', () => {
    expect(isQwenImageEditModel({ variant: null })).toBe(false);
  });

  it('returns false when variant is undefined', () => {
    expect(isQwenImageEditModel({ variant: undefined })).toBe(false);
  });

  it('returns false when variant field is absent', () => {
    expect(isQwenImageEditModel({})).toBe(false);
  });

  it('returns false when model is null', () => {
    expect(isQwenImageEditModel(null)).toBe(false);
  });

  it('returns false for unrelated variant values', () => {
    expect(isQwenImageEditModel({ variant: 'schnell' })).toBe(false);
    expect(isQwenImageEditModel({ variant: 'dev' })).toBe(false);
    expect(isQwenImageEditModel({ variant: 'turbo' })).toBe(false);
  });

  describe('reference image filtering regression', () => {
    it('prevents reference images from leaking to generate models when switching from edit', () => {
      // Simulate: user was using an edit model (variant='edit') with reference images,
      // then switches to a generate model (variant='generate').
      // The generate model should NOT receive reference images.
      const editModel = { variant: 'edit' as const };
      const generateModel = { variant: 'generate' as const };

      // Edit model: reference images should be collected
      expect(isQwenImageEditModel(editModel)).toBe(true);

      // Generate model: reference images must NOT be collected, even if they exist in state
      expect(isQwenImageEditModel(generateModel)).toBe(false);
    });

    it('prevents reference images from leaking to GGUF models without variant', () => {
      // GGUF models installed without a variant field default to generate behavior
      const ggufModelNoVariant = {};
      expect(isQwenImageEditModel(ggufModelNoVariant)).toBe(false);
    });
  });
});
