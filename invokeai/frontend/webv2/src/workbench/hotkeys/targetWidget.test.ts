import type { WidgetPlacementProject } from '@workbench/widgetPlacementCommands';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { getHotkeyTargetWidget, resolveHotkeyTarget } from './targetWidget';

class FakeElement {
  constructor(private readonly attrs: Record<string, string> | null) {}

  closest(): FakeElement | null {
    return this.attrs ? this : null;
  }

  getAttribute(name: string): string | null {
    return this.attrs?.[name] ?? null;
  }
}

describe('getHotkeyTargetWidget', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads the nearest hotkey widget shell from the event target', () => {
    vi.stubGlobal('Element', FakeElement);

    expect(
      getHotkeyTargetWidget(
        new FakeElement({
          'data-hotkey-widget-instance-id': 'preview',
          'data-hotkey-widget-region': 'center',
          'data-hotkey-widget-type-id': 'preview',
        }) as unknown as EventTarget
      )
    ).toEqual({ instanceId: 'preview', region: 'center', typeId: 'preview' });
  });

  it('ignores targets outside widget shells', () => {
    vi.stubGlobal('Element', FakeElement);

    expect(getHotkeyTargetWidget(new FakeElement(null) as unknown as EventTarget)).toBeNull();
  });
});

describe('resolveHotkeyTarget', () => {
  const region = (activeInstanceId: string, instanceIds: string[] = activeInstanceId ? [activeInstanceId] : []) => ({
    activeInstanceId,
    instanceIds,
  });
  const project: WidgetPlacementProject = {
    floatingPlacements: { map: { returnIndex: 1, returnRegion: 'right' } },
    projectId: 'project-1',
    widgetInstances: {
      canvas: { id: 'canvas', typeId: 'canvas' },
      gallery: { id: 'gallery', typeId: 'gallery' },
      map: { id: 'map', typeId: 'image-map' },
    },
    // No `floating` entry exists here, so indexing the regions with a floating target would throw.
    widgetRegions: { bottom: region(''), center: region('canvas'), left: region(''), right: region('gallery') },
  };

  it('aims at the active floating window when the key press lands outside any widget', () => {
    expect(
      resolveHotkeyTarget({ focusTarget: { instanceId: 'map', kind: 'floating' }, project, targetWidget: null })
    ).toEqual({
      activeInstanceId: 'map',
      activeWidgetTypeId: 'image-map',
      focusedRegion: 'floating',
      source: { instanceId: 'map', projectId: 'project-1', region: 'floating', typeId: 'image-map' },
    });
  });

  it('aims at the focused docked region’s active instance when the key press lands outside any widget', () => {
    expect(
      resolveHotkeyTarget({ focusTarget: { kind: 'region', region: 'right' }, project, targetWidget: null })
    ).toEqual({
      activeInstanceId: 'gallery',
      activeWidgetTypeId: 'gallery',
      focusedRegion: 'right',
      source: { instanceId: 'gallery', projectId: 'project-1', region: 'right', typeId: 'gallery' },
    });
  });

  it('prefers the widget under the event target over whatever holds focus', () => {
    const target = resolveHotkeyTarget({
      focusTarget: { instanceId: 'map', kind: 'floating' },
      project,
      targetWidget: { instanceId: 'canvas', region: 'center', typeId: 'canvas' },
    });

    expect(target).toMatchObject({
      activeInstanceId: 'canvas',
      activeWidgetTypeId: 'canvas',
      source: { instanceId: 'canvas', region: 'center' },
    });
  });

  it('gives a key press inside a floating window a floating source', () => {
    const target = resolveHotkeyTarget({
      focusTarget: { instanceId: 'map', kind: 'floating' },
      project,
      targetWidget: { instanceId: 'map', region: 'floating', typeId: 'image-map' },
    });

    expect(target.focusedRegion).toBe('floating');
    expect(target.source).toEqual({
      instanceId: 'map',
      projectId: 'project-1',
      region: 'floating',
      typeId: 'image-map',
    });
  });

  it('ignores a floating focus whose instance no longer floats in this project', () => {
    expect(
      resolveHotkeyTarget({ focusTarget: { instanceId: 'gallery', kind: 'floating' }, project, targetWidget: null })
    ).toEqual({ activeInstanceId: null, activeWidgetTypeId: null, focusedRegion: null, source: null });
  });

  it('has no widget target for a focused region that is empty, or when nothing holds focus', () => {
    expect(
      resolveHotkeyTarget({ focusTarget: { kind: 'region', region: 'left' }, project, targetWidget: null })
    ).toEqual({ activeInstanceId: null, activeWidgetTypeId: null, focusedRegion: 'left', source: null });
    expect(resolveHotkeyTarget({ focusTarget: null, project, targetWidget: null })).toEqual({
      activeInstanceId: null,
      activeWidgetTypeId: null,
      focusedRegion: null,
      source: null,
    });
  });
});
