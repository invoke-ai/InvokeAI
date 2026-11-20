import { describe, expect, it } from 'vitest';

import { getCudaDeviceIndex, getDeviceLabel, getDeviceNameLabels, resolveRandDeviceMetadata } from './deviceLabels';

describe('getCudaDeviceIndex', () => {
  it.each([
    ['cuda:0', 0],
    ['cuda:7', 7],
    ['cuda:12', 12],
    ['xpu:0', 0],
    ['xpu:7', 7],
    ['cpu', null],
    ['mps', null],
    ['cuda', null],
    [null, null],
    [undefined, null],
    ['', null],
    ['cuda:0:1', null],
    ['xcuda:0', null],
  ])('reads %s as index %s', (device, expected) => {
    expect(getCudaDeviceIndex(device)).toBe(expected);
  });
});

describe('resolveRandDeviceMetadata', () => {
  it.each([
    ['cpu whenever CPU noise is enabled', true, [{ device: 'xpu:0', name: 'Arc' }], 'cpu'],
    ['the single cuda family', false, [{ device: 'cuda:0', name: 'RTX' }], 'cuda'],
    ['the single xpu family', false, [{ device: 'xpu:0', name: 'Arc' }], 'xpu'],
    ['the legacy cuda fallback without device data', false, [], 'cuda'],
    [
      'the legacy cuda fallback for mixed families',
      false,
      [
        { device: 'cuda:0', name: 'RTX' },
        { device: 'xpu:0', name: 'Arc' },
      ],
      'cuda',
    ],
  ])('reports %s', (_label, cpuNoise, devices, expected) => {
    expect(resolveRandDeviceMetadata(cpuNoise, devices)).toBe(expected);
  });
});

describe('getDeviceNameLabels', () => {
  it('suffixes only the names that repeat', () => {
    expect(
      getDeviceNameLabels([
        { device: 'cuda:0', name: 'RTX 5090' },
        { device: 'cuda:1', name: 'RTX 4090' },
        { device: 'cuda:2', name: 'RTX 5090' },
      ])
    ).toEqual({ 'cuda:0': 'RTX 5090 #1', 'cuda:1': 'RTX 4090', 'cuda:2': 'RTX 5090 #2' });
  });

  it('numbers by position in the full device set, so disabling a device does not renumber it', () => {
    // Device numbering follows all reported accelerators; disabling one must not renumber the others.
    const allDevices = [
      { device: 'cuda:0', name: 'RTX 5090' },
      { device: 'cuda:1', name: 'RTX 5090' },
      { device: 'cuda:2', name: 'RTX 5090' },
    ];

    expect(getDeviceNameLabels(allDevices)['cuda:2']).toBe('RTX 5090 #3');
  });
});

describe('getDeviceLabel', () => {
  const twoGpus = [
    { device: 'cuda:0', name: 'RTX 5090' },
    { device: 'cuda:1', name: 'RTX 4090' },
  ];

  it('returns the index and name for a known cuda device', () => {
    expect(getDeviceLabel('cuda:1', twoGpus)).toEqual({ index: 1, name: 'RTX 4090' });
  });

  it('returns null on a single-GPU install, where a badge is only noise', () => {
    expect(getDeviceLabel('cuda:0', [{ device: 'cuda:0', name: 'RTX 5090' }])).toBeNull();
  });

  it('returns null for non-cuda, missing, and unreported devices', () => {
    expect(getDeviceLabel('cpu', twoGpus)).toBeNull();
    expect(getDeviceLabel(null, twoGpus)).toBeNull();
    expect(getDeviceLabel(undefined, twoGpus)).toBeNull();
    expect(getDeviceLabel('cuda:9', twoGpus)).toBeNull();
  });

  it('returns null when no options have loaded yet', () => {
    expect(getDeviceLabel('cuda:0', [])).toBeNull();
  });
});
