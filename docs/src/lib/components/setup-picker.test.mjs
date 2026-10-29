import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import en from '../../content/i18n/en.json' with { type: 'json' };
import {
  describeVariant,
  findCoverageGaps,
  localizeSetupOptions,
  resolveSetup,
  SETUP_OPTIONS,
  variantsInHtml,
} from './setup-picker.mjs';

// The English page's options.
const options = localizeSetupOptions(SETUP_OPTIONS, (key) => en[key]);

describe('resolveSetup', () => {
  it('keeps a valid stored setup regardless of the platform', () => {
    assert.deepEqual(resolveSetup(options, { os: 'linux', gpu: 'amd' }, 'macOS'), { os: 'linux', gpu: 'amd' });
    assert.deepEqual(resolveSetup(options, { os: 'windows', gpu: 'arm64' }, 'Linux'), {
      os: 'windows',
      gpu: 'arm64',
    });
  });

  it("falls back to the OS's default GPU when the OS does not offer the GPU", () => {
    assert.deepEqual(resolveSetup(options, { os: 'macos', gpu: 'nvidia' }), { os: 'macos', gpu: 'apple' });
    assert.deepEqual(resolveSetup(options, { os: 'windows', gpu: 'apple' }), { os: 'windows', gpu: 'nvidia' });
    assert.deepEqual(resolveSetup(options, { os: 'linux', gpu: 'arm64' }), { os: 'linux', gpu: 'nvidia' });
  });

  it('detects the OS from the platform on a first visit', () => {
    assert.deepEqual(resolveSetup(options, null, 'MacIntel'), { os: 'macos', gpu: 'apple' });
    assert.deepEqual(resolveSetup(options, null, 'macOS'), { os: 'macos', gpu: 'apple' });
    assert.deepEqual(resolveSetup(options, null, 'Linux x86_64'), { os: 'linux', gpu: 'nvidia' });
    assert.deepEqual(resolveSetup(options, null, 'Win32'), { os: 'windows', gpu: 'nvidia' });
    assert.deepEqual(resolveSetup(options, null, undefined), { os: 'windows', gpu: 'nvidia' });
  });

  it('ignores stored junk and unknown ids', () => {
    for (const junk of ['macos', 42, [], { os: 'beos' }]) {
      assert.deepEqual(resolveSetup(options, junk, 'Linux'), { os: 'linux', gpu: 'nvidia' });
    }
    assert.deepEqual(resolveSetup(options, { os: 'beos', gpu: 'amd' }, 'Linux'), { os: 'linux', gpu: 'amd' });
  });
});

describe('describeVariant', () => {
  it('labels a variant with its OS and GPU names', () => {
    assert.deepEqual(describeVariant(options, { os: 'windows linux', gpu: 'nvidia' }), {
      os: ['windows', 'linux'],
      gpu: ['nvidia'],
      label: 'Windows, Linux · NVIDIA',
    });
    assert.equal(describeVariant(options, { os: 'linux', gpu: 'none' }).label, 'Linux · None (CPU)');
    assert.deepEqual(describeVariant(options, { os: 'macos' }), { os: ['macos'], gpu: undefined, label: 'macOS' });
  });

  it('rejects typos and GPUs the chosen OS does not offer, so no command is silently hidden', () => {
    assert.throws(() => describeVariant(options, { os: 'window' }), /Unknown setup OS "window"/);
    assert.throws(() => describeVariant(options, { gpu: 'nvida' }), /GPU "nvida"/);
    assert.throws(() => describeVariant(options, { os: 'macos', gpu: 'nvidia' }), /GPU "nvidia"/);
    assert.throws(() => describeVariant(options, {}), /needs an `os` or a `gpu`/);
  });
});

describe('findCoverageGaps', () => {
  it('accepts alternatives that show exactly one variant for every setup', () => {
    const variants = [
      { os: ['windows', 'linux'], gpu: ['nvidia'] },
      { gpu: ['amd', 'intel', 'none', 'apple'] },
      { os: ['windows'], gpu: ['arm64'] },
    ];
    assert.deepEqual(findCoverageGaps(options, variants), { missing: [], repeated: [] });
  });

  it('names setups that see no variant or more than one', () => {
    const variants = [{ os: ['windows'] }, { os: ['windows', 'linux'], gpu: ['nvidia'] }];
    assert.deepEqual(findCoverageGaps(options, variants), {
      missing: ['macOS · Apple silicon', 'Linux · AMD', 'Linux · Intel Arc', 'Linux · None (CPU)'],
      repeated: ['Windows · NVIDIA'],
    });
  });
});

describe('variantsInHtml', () => {
  it('reads the ids of rendered variants, treating a missing attribute as any', () => {
    const html =
      '<div class="setup-variant" data-setup-variant data-setup-for-os="windows linux" data-setup-for-gpu="amd">' +
      '<p>x</p></div><div data-setup-variant class="setup-variant" data-setup-for-gpu="arm64"></div>' +
      '<div class="other" data-setup-for-os="macos"></div>';
    assert.deepEqual(variantsInHtml(html), [
      { os: ['windows', 'linux'], gpu: ['amd'] },
      { os: undefined, gpu: ['arm64'] },
    ]);
  });
});
