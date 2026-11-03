import type { RegisteredWidget } from '@workbench/widgetContracts';

import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { describe, expect, it, vi } from 'vitest';

import { canAddWidget } from './widgetsListModel';

describe('canAddWidget', () => {
  // Every row asks on every render; starting a chunk load there pulled whole widget bundles in with the list.
  it('answers for panels and status chips without loading any widget', () => {
    const preloads = vi.fn();
    const quiet = (widget: RegisteredWidget): RegisteredWidget => ({
      ...widget,
      implementation: { ...widget.implementation, preload: preloads },
    });
    const regions = (region: Parameters<typeof getWidgetsForRegion>[0]) => getWidgetsForRegion(region).map(quiet);
    const byId = (typeId: string) =>
      regions('center')
        .concat(regions('right'), regions('left'), regions('bottom'))
        .find((widget) => widget.manifest.id === typeId)!;

    expect(canAddWidget(byId('queue'), regions)).toBe(true);
    expect(canAddWidget(byId('autosave-status'), regions)).toBe(true);
    expect(preloads).not.toHaveBeenCalled();
  });
});
