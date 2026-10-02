import { graphWidgetSources } from '@workbench/graphWidgets';
import { describe, expect, it } from 'vitest';

import { getInitialLayoutPresetIconId, getInitialLayoutPresetRoute } from './layoutPresetDialogModel';

describe('layout preset dialog model', () => {
  it('falls back to a focusable icon when a persisted icon id is unknown', () => {
    expect(getInitialLayoutPresetIconId('star')).toBe('star');
    expect(getInitialLayoutPresetIconId('retired-icon')).toBe('layout-grid');
    expect(getInitialLayoutPresetIconId(undefined)).toBe('layout-grid');
  });

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
