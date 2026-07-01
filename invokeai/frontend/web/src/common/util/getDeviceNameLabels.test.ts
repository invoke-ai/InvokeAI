import type { S } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { getDeviceNameLabels } from './getDeviceNameLabels';

const opt = (device: string, name: string): S['GenerationDeviceOption'] => ({ device, name });

describe('getDeviceNameLabels', () => {
  it('adds a 1-based #N suffix to identically-named devices', () => {
    const labels = getDeviceNameLabels([opt('cuda:0', 'AMD Radeon PRO W7900'), opt('cuda:1', 'AMD Radeon PRO W7900')]);
    expect(labels).toEqual({
      'cuda:0': 'AMD Radeon PRO W7900 #1',
      'cuda:1': 'AMD Radeon PRO W7900 #2',
    });
  });

  it('does not add a suffix to a uniquely-named device', () => {
    const labels = getDeviceNameLabels([opt('cuda:0', 'AMD Radeon PRO W7900')]);
    expect(labels).toEqual({ 'cuda:0': 'AMD Radeon PRO W7900' });
  });

  it('only suffixes the names that are duplicated', () => {
    const labels = getDeviceNameLabels([
      opt('cuda:0', 'RTX 4090'),
      opt('cuda:1', 'RTX 3090'),
      opt('cuda:2', 'RTX 3090'),
    ]);
    expect(labels).toEqual({
      'cuda:0': 'RTX 4090',
      'cuda:1': 'RTX 3090 #1',
      'cuda:2': 'RTX 3090 #2',
    });
  });

  it('returns an empty map for no options', () => {
    expect(getDeviceNameLabels([])).toEqual({});
  });
});
