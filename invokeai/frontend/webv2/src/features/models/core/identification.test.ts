import type { ModelConfig } from '@features/models/core/types';

import { describe, expect, it } from 'vitest';

import { chunkKeys, getIdentityChanges, shouldAutoIdentify } from './identification';

const model = (fields: Partial<ModelConfig>) =>
  ({
    base: 'sd-1',
    format: 'checkpoint',
    path: 'a/model.safetensors',
    type: 'main',
    variant: 'normal',
    ...fields,
  }) as ModelConfig;

describe('shouldAutoIdentify', () => {
  it.each([
    ['an Unknown record whatever its file', { path: 'x.ckpt', type: 'unknown' }, true],
    ['a safetensors file', { path: 'A/Model.SafeTensors' }, true],
    ['a GGUF file', { path: 'model.gguf' }, true],
    ['a GGUF-format record', { format: 'gguf', path: 'model' }, true],
    ['a pickled checkpoint', { path: 'model.ckpt' }, false],
    ['a .pt file', { path: 'model.pt' }, false],
    ['a diffusers folder', { format: 'diffusers', path: 'folder' }, false],
  ] as const)('%s', (_name, fields, expected) => {
    expect(shouldAutoIdentify(model(fields as Partial<ModelConfig>))).toBe(expected);
  });
});

describe('getIdentityChanges', () => {
  it('lists only the fields that differ, with a missing variant as null', () => {
    expect(
      getIdentityChanges(model({ base: 'sd-1', variant: null }), model({ base: 'sdxl', variant: 'inpaint' }))
    ).toEqual([
      { field: 'base', from: 'SD 1.x', to: 'SDXL' },
      { field: 'variant', from: null, to: 'Inpaint' },
    ]);
    expect(getIdentityChanges(model({}), model({}))).toEqual([]);
  });
});

describe('chunkKeys', () => {
  it('splits into ordered batches of at most the size', () => {
    const keys = Array.from({ length: 250 }, (_, index) => `k${index}`);

    expect(chunkKeys(keys, 100).map((chunk) => chunk.length)).toEqual([100, 100, 50]);
    expect(chunkKeys(keys)[0]).toHaveLength(10);
    expect(chunkKeys(keys).flat()).toEqual(keys);
    expect(chunkKeys([])).toEqual([]);
  });
});
