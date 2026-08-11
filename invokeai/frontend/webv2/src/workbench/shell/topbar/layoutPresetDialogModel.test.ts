import { graphWidgetSources } from '@workbench/graphWidgets';
import { describe, expect, it } from 'vitest';

import { getInitialLayoutPresetRoute } from './layoutPresetDialogModel';

describe('layout preset dialog model', () => {
  it('keeps an available saved route and repairs a stale source from the first available option', () => {
    const options = graphWidgetSources.filter(
      (source) => source.sourceId === 'generate' || source.sourceId === 'canvas'
    );

    expect(getInitialLayoutPresetRoute({ destination: 'canvas', sourceId: 'canvas' }, options)).toEqual({
      destination: 'canvas',
      sourceId: 'canvas',
    });
    expect(getInitialLayoutPresetRoute({ destination: 'gallery', sourceId: 'workflow' }, options)).toEqual({
      destination: 'gallery',
      sourceId: 'generate',
    });
    expect(getInitialLayoutPresetRoute({ destination: 'gallery', sourceId: 'workflow' }, [])).toBeUndefined();
  });
});
