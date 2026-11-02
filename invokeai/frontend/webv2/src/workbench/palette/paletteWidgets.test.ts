import type { RegisteredWidget } from '@workbench/widgetContracts';

import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { describe, expect, it, vi } from 'vitest';

import { listPaletteWidgets } from './paletteWidgets';

describe('listPaletteWidgets', () => {
  // The palette builds this list on every open; only the entry the user runs may load its widget.
  it('lists each openable widget once, loading none of them', () => {
    const preload = vi.fn();
    const quiet = (widget: RegisteredWidget): RegisteredWidget => ({
      ...widget,
      implementation: { ...widget.implementation, preload },
    });

    const ids = listPaletteWidgets((region) => getWidgetsForRegion(region).map(quiet)).map(
      ({ manifest }) => manifest.id
    );

    expect(preload).not.toHaveBeenCalled();
    expect(ids).toEqual(expect.arrayContaining(['canvas', 'gallery', 'generate', 'queue', 'upscale']));
    expect(new Set(ids).size).toBe(ids.length);
    // Status chips render no panel, and the Workflow editor is offered through its own command.
    expect(ids).not.toContain('autosave-status');
    expect(ids).not.toContain('workflow');
  });
});
