import { describe, expect, it } from 'vitest';

import { getLayerFilterControlPolicy } from './LayerFilterControls';

describe('getLayerFilterControlPolicy', () => {
  it('keeps compact layer properties while giving operation controls practical targets', () => {
    expect(getLayerFilterControlPolicy('property')).toEqual({
      controlMinH: undefined,
      controlSize: 'xs',
      modelSize: 'xs',
    });
    expect(getLayerFilterControlPolicy('operation')).toEqual({ controlMinH: '10', controlSize: 'md', modelSize: 'md' });
  });
});
