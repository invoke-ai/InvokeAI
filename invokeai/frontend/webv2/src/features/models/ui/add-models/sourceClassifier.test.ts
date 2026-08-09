import { describe, expect, it } from 'vitest';

import { classifySource } from './sourceClassifier';

describe('classifySource', () => {
  it.each([
    ['https://example.com/model.safetensors', 'URL'],
    ['HTTP://EXAMPLE.COM/model', 'URL'],
    ['http://civitai.com/api/download/models/12345', 'URL'],
  ])('classifies %s as a URL', (value, label) => {
    expect(classifySource(value)).toMatchObject({ isInstallable: true, label, localKind: null, looksUrl: true });
  });

  it.each([
    'black-forest-labs/FLUX.1-dev',
    'owner/repo:fp16',
    'owner/repo:fp16:path/to/file.safetensors',
    'owner.name/repo-name',
  ])('classifies %s as a HuggingFace repo', (value) => {
    expect(classifySource(value)).toMatchObject({ isInstallable: true, label: 'Hugging Face repo', looksRepo: true });
  });

  it.each([
    ['/models/sdxl.safetensors', 'file'],
    ['C:\\models\\sdxl.safetensors', 'file'],
    ['/models/checkpoints/', 'folder'],
    ['C:/models/checkpoints', 'folder'],
    ['/models/no-extension', 'folder'],
  ])('classifies %s as a local %s', (value, localKind) => {
    expect(classifySource(value)).toMatchObject({ isInstallable: true, localKind, looksLocal: true });
  });

  it.each(['juggernaut', 'flux dev', 'owner/repo with spaces', ''])('treats %j as a search term', (value) => {
    expect(classifySource(value)).toMatchObject({ isInstallable: false, label: null, localKind: null });
  });
});
