import { describe, expect, it } from 'vitest';

import { curvePointFromSvg, curvePointToSvg } from './curveEditorMath';

describe('curve editor coordinates', () => {
  it('insets endpoints so handles are fully visible', () => {
    expect(curvePointToSvg(0, 0)).toEqual({ cx: 6, cy: 174 });
    expect(curvePointToSvg(255, 255)).toEqual({ cx: 174, cy: 6 });
  });

  it('round-trips and clamps pointer coordinates through the inset plot', () => {
    expect(curvePointFromSvg(90, 90)).toEqual([128, 128]);
    expect(curvePointFromSvg(-20, 220)).toEqual([0, 0]);
    expect(curvePointFromSvg(220, -20)).toEqual([255, 255]);
  });
});
