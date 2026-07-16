import { describe, expect, it } from 'vitest';

import { getLayerFilterControlPolicy } from './LayerFilterControls';

describe('getLayerFilterControlPolicy', () => {
  it('keeps layer properties compact and gives the operation bar fixed-width upward-opening fields', () => {
    expect(getLayerFilterControlPolicy('property')).toEqual({
      controlMinH: undefined,
      controlSize: 'xs',
      fieldOrientation: 'vertical',
      fieldW: undefined,
      modelSize: 'xs',
      positioning: { placement: 'bottom-end', sameWidth: false },
      showFilterLabel: true,
      showNumberStepper: true,
    });
    expect(getLayerFilterControlPolicy('operation')).toEqual({
      controlMinH: undefined,
      controlSize: 'xs',
      fieldOrientation: 'horizontal',
      fieldW: { enum: '13rem', filter: '11rem', model: '16rem', number: '17rem', string: '13rem' },
      modelSize: 'xs',
      positioning: { placement: 'top-end', sameWidth: false },
      showFilterLabel: false,
      showNumberStepper: false,
    });
  });
});
