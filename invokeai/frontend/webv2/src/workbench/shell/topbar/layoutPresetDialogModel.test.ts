import { getGraphWidgetSources } from '@workbench/graphWidgets';
import { describe, expect, it } from 'vitest';

import {
  getInitialLayoutPresetIconId,
  getInitialLayoutPresetRoute,
  resolveLayoutPresetRoute,
} from './layoutPresetDialogModel';

describe('workspace route and icon model', () => {
  it('falls back to a focusable icon when a persisted icon id is unknown', () => {
    expect(getInitialLayoutPresetIconId('star')).toBe('star');
    expect(getInitialLayoutPresetIconId('retired-icon')).toBe('layout-grid');
    expect(getInitialLayoutPresetIconId(undefined)).toBe('layout-grid');
  });

  it('keeps an available saved route and repairs a stale source from the first available option', () => {
    const options = getGraphWidgetSources().filter(
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

  it('keeps Auto only where a workspace may go without a route', () => {
    const options = getGraphWidgetSources().filter((source) => source.sourceId === 'generate');

    expect(resolveLayoutPresetRoute(null, options, true)).toBeNull();
    expect(resolveLayoutPresetRoute(null, options, false)).toEqual({ destination: 'gallery', sourceId: 'generate' });
  });
});
