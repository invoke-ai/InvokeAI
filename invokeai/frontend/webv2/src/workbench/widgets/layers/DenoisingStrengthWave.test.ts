import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { DenoisingStrengthWave, getDenoisingStrengthWavePath } from './DenoisingStrengthWave';

describe('getDenoisingStrengthWavePath', () => {
  it('renders zero strength as a straight centered line', () => {
    expect(getDenoisingStrengthWavePath(0, 100, 14, 5)).toBe('M0,7 L100,7');
  });

  it('increases wave amplitude with denoising strength', () => {
    const low = getDenoisingStrengthWavePath(0.25, 100, 14, 5);
    const high = getDenoisingStrengthWavePath(1, 100, 14, 5);

    expect(low).toContain('Q10,4.5 20,7');
    expect(high).toContain('Q10,0 20,7');
  });

  it('clamps strength before calculating amplitude', () => {
    expect(getDenoisingStrengthWavePath(-1, 100, 14, 5)).toBe(getDenoisingStrengthWavePath(0, 100, 14, 5));
    expect(getDenoisingStrengthWavePath(2, 100, 14, 5)).toBe(getDenoisingStrengthWavePath(1, 100, 14, 5));
  });

  it('renders one continuous five-segment stroke without value-based dashes', () => {
    const markup = renderToStaticMarkup(createElement(DenoisingStrengthWave, { value: 0.52 }));

    expect(markup.match(/<path/g)).toHaveLength(1);
    expect(markup).not.toContain('stroke-dasharray');
    expect(markup.match(/ Q/g)).toHaveLength(5);
    expect(markup).toContain('width="56"');
    expect(markup).not.toContain('position:absolute');
  });
});
