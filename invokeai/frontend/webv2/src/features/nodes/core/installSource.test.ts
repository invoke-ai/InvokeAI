import { describe, expect, it } from 'vitest';

import { derivePackNameFromSource, validateInstallSource } from './installSource';

describe('derivePackNameFromSource', () => {
  it.each([
    ['https://github.com/owner/my-pack.git', 'my-pack'],
    ['https://github.com/owner/my-pack', 'my-pack'],
    ['https://github.com/owner/my-pack/', 'my-pack'],
    ['https://github.com/owner/my-pack.git///', 'my-pack'],
    ['git@host:x/pack_1.2.git', 'pack_1.2'],
    ['plain-name', 'plain-name'],
  ])('derives %s -> %s', (source, expected) => {
    expect(derivePackNameFromSource(source)).toBe(expected);
  });

  it.each([
    'https://github.com/owner/..',
    'https://github.com/owner/.',
    'https://github.com/owner/.hidden',
    'https://github.com/owner/-dash-first',
    'https://x/pack name',
    'https://x/pack\\name',
    '',
    '///',
  ])('rejects %j', (source) => {
    expect(derivePackNameFromSource(source)).toBeNull();
  });
});

describe('validateInstallSource', () => {
  const installed = new Set(['existing-pack']);

  it('classifies empty, invalid, installed, and installable sources', () => {
    expect(validateInstallSource('   ', installed)).toEqual({ issue: 'empty', packName: null });
    expect(validateInstallSource('https://x/..', installed)).toEqual({ issue: 'invalidPackName', packName: null });
    expect(validateInstallSource('https://x/existing-pack.git', installed)).toEqual({
      issue: 'alreadyInstalled',
      packName: 'existing-pack',
    });
    expect(validateInstallSource('https://x/new-pack.git', installed)).toEqual({
      issue: null,
      packName: 'new-pack',
    });
  });
});
