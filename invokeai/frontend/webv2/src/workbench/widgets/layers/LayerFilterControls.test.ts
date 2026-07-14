import { describe, expect, it } from 'vitest';

import { getLayerFilterControlPolicy } from './LayerFilterControls';

describe('getLayerFilterControlPolicy', () => {
  it('keeps layer properties compact and gives the operation bar fixed-width upward-opening fields', () => {
    expect(getLayerFilterControlPolicy('property')).toEqual({
      controlMinH: undefined,
      controlSize: 'xs',
      fieldW: undefined,
      modelSize: 'xs',
      positioning: { placement: 'bottom-end', sameWidth: false },
    });
    expect(getLayerFilterControlPolicy('operation')).toEqual({
      controlMinH: undefined,
      controlSize: 'xs',
      fieldW: { enum: '9rem', filter: '11rem', model: '14rem', number: '13rem', string: '9rem' },
      modelSize: 'xs',
      positioning: { placement: 'top-end', sameWidth: false },
    });
  });
});
