import { accountLifecycle } from '@platform/state/accountLifecycle';
import { expect, it } from 'vitest';

import { advanceMaskFillColors, nextMaskFillColor } from './maskFillPalette';

it('cycles mask colors independently in palette order and restarts at the session defaults', () => {
  accountLifecycle.invalidate();
  expect(nextMaskFillColor('inpaint_mask')).toBe('#e07575');
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  advanceMaskFillColors('inpaint_mask');
  expect(Array.from({ length: 10 }, (_, index) => nextMaskFillColor('inpaint_mask', index))).toEqual([
    '#83d683',
    '#F59E42',
    '#F4D35E',
    '#38C9C5',
    '#62B5F5',
    '#637BE8',
    '#AD80E8',
    '#ED79BA',
    '#B7D65A',
    '#e07575',
  ]);
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  advanceMaskFillColors('regional_guidance');
  expect(nextMaskFillColor('regional_guidance')).toBe('#F59E42');
  accountLifecycle.invalidate();
  expect(nextMaskFillColor('regional_guidance')).toBe('#83d683');
  expect(nextMaskFillColor('inpaint_mask')).toBe('#e07575');
  accountLifecycle.invalidate();
});
