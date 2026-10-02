import { describe, expect, it } from 'vitest';

import { getLayerFilterControlPolicy } from './LayerFilterControls';

describe('getLayerFilterControlPolicy', () => {
  it('renders compact vertical fields with steppers and downward-opening selects', () => {
    expect(getLayerFilterControlPolicy()).toEqual({
      controlMinH: undefined,
      controlSize: 'xs',
      fieldOrientation: 'vertical',
      modelSize: 'xs',
      positioning: { placement: 'bottom-end', sameWidth: false },
      showFilterLabel: true,
      showNumberStepper: true,
    });
  });
});
