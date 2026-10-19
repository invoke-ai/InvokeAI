import { describe, expect, it } from 'vitest';

import { classifySource } from './sourceClassifier';

describe('classifySource', () => {
  it.each([
    'https://example.com/model.safetensors',
    'HTTP://EXAMPLE.COM/model',
    'http://civitai.com/api/download/models/12345',
  ])('classifies %s as a URL', (value) => {
    expect(classifySource(value)).toMatchObject({
      isInstallable: true,
      labelKey: 'models.sourceKind.url',
      localKind: null,
      looksUrl: true,
    });
  });

  it.each([
    'black-forest-labs/FLUX.1-dev',
    'owner/repo:fp16',
    'owner/repo:fp16:path/to/file.safetensors',
    'owner.name/repo-name',
    'stabilityai/stable-diffusion-2.1',
  ])('classifies %s as a HuggingFace repo', (value) => {
    expect(classifySource(value)).toMatchObject({
      isInstallable: true,
      labelKey: 'models.sourceKind.hfRepo',
      looksRepo: true,
    });
  });

  it.each([
    ['/models/sdxl.safetensors', 'file'],
    ['C:\\models\\sdxl.safetensors', 'file'],
    ['/models/checkpoints/', 'folder'],
    ['C:/models/checkpoints', 'folder'],
    ['/models/no-extension', 'folder'],
  ])('classifies %s as a local %s', (value, localKind) => {
    expect(classifySource(value)).toMatchObject({
      isInstallable: true,
      labelKey: `models.sourceKind.${localKind}Path`,
      localKind,
      looksLocal: true,
    });
  });

  it.each(['juggernaut', 'flux dev', 'owner/repo with spaces', ''])('treats %j as a search term', (value) => {
    expect(classifySource(value)).toMatchObject({ isInstallable: false, labelKey: null, localKind: null });
  });

  // Relative paths satisfy the repo shape but are not installable sources.
  it.each(['models/juggernaut.safetensors', 'checkpoints/model.ckpt', './owner/repo', '../models/foo.gguf'])(
    'treats the relative path %s as a search term, not a repo',
    (value) => {
      expect(classifySource(value)).toMatchObject({
        isInstallable: false,
        labelKey: null,
        localKind: null,
        looksRepo: false,
      });
    }
  );
});
