import { describe, expect, it } from 'vitest';

import { sourceLocation } from './useSourceNameFilter';

describe('sourceLocation', () => {
  it('keeps the part of a Hugging Face file URL inside the repo', () => {
    expect(sourceLocation('https://huggingface.co/org/repo/resolve/main/v2/model.safetensors')).toBe(
      'v2/model.safetensors'
    );
  });

  it('keeps the part of a scanned path below the scan root, with or without a trailing separator', () => {
    expect(sourceLocation('/mnt/models/loras/a/model.safetensors', '/mnt/models')).toBe('loras/a/model.safetensors');
    expect(sourceLocation('/mnt/models/loras/a/model.safetensors', '/mnt/models/')).toBe('loras/a/model.safetensors');
  });

  it('adds nothing when the source sits directly in its root', () => {
    expect(sourceLocation('https://huggingface.co/org/repo/resolve/main/model.safetensors')).toBeNull();
    expect(sourceLocation('/mnt/models/model.safetensors', '/mnt/models')).toBeNull();
  });
});
