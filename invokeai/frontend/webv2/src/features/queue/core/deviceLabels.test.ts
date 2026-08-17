import { describe, expect, it } from 'vitest';

import { getCudaDeviceIndex, getDeviceLabel, getDeviceNameLabels, resolveRandDeviceMetadata } from './deviceLabels';

describe('getCudaDeviceIndex', () => {
  it('parses the index from a cuda device', () => {
    expect(getCudaDeviceIndex('cuda:0')).toBe(0);
    expect(getCudaDeviceIndex('cuda:7')).toBe(7);
    expect(getCudaDeviceIndex('cuda:12')).toBe(12);
  });

  it('parses the index from an xpu device', () => {
    expect(getCudaDeviceIndex('xpu:0')).toBe(0);
    expect(getCudaDeviceIndex('xpu:7')).toBe(7);
  });

  it('returns null for non-cuda and missing devices', () => {
    expect(getCudaDeviceIndex('cpu')).toBeNull();
    expect(getCudaDeviceIndex('mps')).toBeNull();
    expect(getCudaDeviceIndex('cuda')).toBeNull();
    expect(getCudaDeviceIndex(null)).toBeNull();
    expect(getCudaDeviceIndex(undefined)).toBeNull();
    expect(getCudaDeviceIndex('')).toBeNull();
  });

  it('does not accept a device with trailing content', () => {
    expect(getCudaDeviceIndex('cuda:0:1')).toBeNull();
    expect(getCudaDeviceIndex('xcuda:0')).toBeNull();
  });
});

describe('resolveRandDeviceMetadata', () => {
  it('always reports CPU when CPU noise is enabled', () => {
    expect(resolveRandDeviceMetadata(true, [{ device: 'xpu:0', name: 'Arc' }])).toBe('cpu');
  });

  it('reports the single available accelerator family', () => {
    expect(resolveRandDeviceMetadata(false, [{ device: 'cuda:0', name: 'RTX' }])).toBe('cuda');
    expect(resolveRandDeviceMetadata(false, [{ device: 'xpu:0', name: 'Arc' }])).toBe('xpu');
  });

  it('keeps the legacy CUDA fallback when device data is missing or mixed', () => {
    expect(resolveRandDeviceMetadata(false, [])).toBe('cuda');
    expect(
      resolveRandDeviceMetadata(false, [
        { device: 'cuda:0', name: 'RTX' },
        { device: 'xpu:0', name: 'Arc' },
      ])
    ).toBe('cuda');
  });
});

describe('getDeviceNameLabels', () => {
  it('leaves a uniquely-named device unsuffixed', () => {
    expect(getDeviceNameLabels([{ device: 'cuda:0', name: 'RTX 5090' }])).toEqual({ 'cuda:0': 'RTX 5090' });
  });

  it('suffixes identically-named devices with a 1-based ordinal', () => {
    expect(
      getDeviceNameLabels([
        { device: 'cuda:0', name: 'RTX 5090' },
        { device: 'cuda:1', name: 'RTX 5090' },
      ])
    ).toEqual({ 'cuda:0': 'RTX 5090 #1', 'cuda:1': 'RTX 5090 #2' });
  });

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
    // The backend reports every installed CUDA device regardless of `generation_devices`, so
    // cuda:2 stays #3 whether or not cuda:1 is enabled for generation. Numbering off a filtered
    // list would silently promote cuda:2 to #2 and disagree with the backend's startup log.
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
