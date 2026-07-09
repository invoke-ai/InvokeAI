import { describe, expect, it } from 'vitest';

import { getDenoisingStrengthWavePath } from './DenoisingStrengthWave';

describe('getDenoisingStrengthWavePath', () => {
  it('renders zero strength as a straight centered line', () => {
    expect(getDenoisingStrengthWavePath(0, 100, 12, 10)).toBe('M0,6 L100,6');
  });

  it('increases wave amplitude with denoising strength', () => {
    const low = getDenoisingStrengthWavePath(0.25, 100, 12, 10);
    const high = getDenoisingStrengthWavePath(1, 100, 12, 10);

    expect(low).toContain('Q5,4.75 10,6');
    expect(high).toContain('Q5,1 10,6');
  });

  it('clamps strength before calculating amplitude', () => {
    expect(getDenoisingStrengthWavePath(-1, 100, 12, 10)).toBe(getDenoisingStrengthWavePath(0, 100, 12, 10));
    expect(getDenoisingStrengthWavePath(2, 100, 12, 10)).toBe(getDenoisingStrengthWavePath(1, 100, 12, 10));
  });
});
