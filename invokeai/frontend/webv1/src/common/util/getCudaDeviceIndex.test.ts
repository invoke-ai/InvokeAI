import { describe, expect, it } from 'vitest';

import { getCudaDeviceIndex } from './getCudaDeviceIndex';

describe('getCudaDeviceIndex', () => {
  it('parses the index from a cuda device string', () => {
    expect(getCudaDeviceIndex('cuda:0')).toBe(0);
    expect(getCudaDeviceIndex('cuda:1')).toBe(1);
    expect(getCudaDeviceIndex('cuda:11')).toBe(11);
  });

  it('parses the index from an xpu device string', () => {
    expect(getCudaDeviceIndex('xpu:0')).toBe(0);
    expect(getCudaDeviceIndex('xpu:1')).toBe(1);
    expect(getCudaDeviceIndex('xpu:11')).toBe(11);
  });

  it('returns null for non-gpu devices', () => {
    expect(getCudaDeviceIndex('cpu')).toBeNull();
    expect(getCudaDeviceIndex('mps')).toBeNull();
  });

  it('returns null for null/undefined/empty', () => {
    expect(getCudaDeviceIndex(null)).toBeNull();
    expect(getCudaDeviceIndex(undefined)).toBeNull();
    expect(getCudaDeviceIndex('')).toBeNull();
  });

  it('returns null for malformed device strings', () => {
    expect(getCudaDeviceIndex('cuda')).toBeNull();
    expect(getCudaDeviceIndex('cuda:')).toBeNull();
    expect(getCudaDeviceIndex('cuda:x')).toBeNull();
    expect(getCudaDeviceIndex('cuda:0:0')).toBeNull();
    expect(getCudaDeviceIndex('xpu')).toBeNull();
    expect(getCudaDeviceIndex('xpu:x')).toBeNull();
  });
});
