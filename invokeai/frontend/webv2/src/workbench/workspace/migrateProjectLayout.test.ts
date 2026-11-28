import type { FloatingWidgetState, PanelState, WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';

import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { getLayoutPreset, layoutPresets } from '@workbench/layoutPresets';
import { describe, expect, it } from 'vitest';

import type { StoredProjectLayout } from './migrateProjectLayout';
import type { ResolvedLayout } from './resolveLayout';

import { registryAvailability } from './layoutFixtures.testing';
import { migrateProjectLayout } from './migrateProjectLayout';
import { resolveLayout } from './resolveLayout';
import { storedOn, structuralLayouts, viewOnlyLayouts, workingCopyOf } from './storedLayoutFixtures.testing';
import { clampRegionSize, PANEL_OPEN_KEYS } from './structure';
import { pruneView } from './view';

const workspaces = Object.fromEntries(layoutPresets.map(({ id, snapshot }) => [id, snapshot]));
const migrate = (stored: StoredProjectLayout) =>
  migrateProjectLayout(stored, { defaultWorkspaceId: 'compose', surfaces: registryAvailability, workspaces });

interface Arrangement {
  floatingWidgets?: Record<string, FloatingWidgetState>;
  layout: { panels: PanelState };
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
}

/** What the shell shows for an arrangement: each region's front, members, whether it is shown, and its size. */
const onScreen = ({ floatingWidgets, layout, widgetRegions }: Arrangement) => ({
  regions: Object.fromEntries(
    WIDGET_REGIONS.map((region) => {
      const { activeInstanceId, instanceIds, isCollapsed, sizePx } = widgetRegions[region];

      return [
        region,
        region === 'center'
          ? { activeInstanceId, instanceIds }
          : {
              activeInstanceId,
              instanceIds,
              isShown: !isCollapsed && layout.panels[PANEL_OPEN_KEYS[region]],
              sizePx: clampRegionSize(region, sizePx),
            },
      ];
    })
  ),
  windows: floatingWidgets ?? {},
});

const resolvedScreen = (layout: ResolvedLayout) => onScreen({ ...layout, layout: { panels: layout.panels } });

/** Every arrangement a stored project holds, by the workspace id it migrates to. */
const arrangementsOf = (stored: StoredProjectLayout, workspaceId: string): [string, Arrangement][] => [
  [workspaceId, stored],
  ...(stored.presetWorkingLayouts ?? []).map(({ presetId, snapshot }): [string, Arrangement] => [presetId, snapshot]),
];

const viewOnlyFixtures = viewOnlyLayouts;
const structuralFixtures = structuralLayouts;

describe('migrateProjectLayout', () => {
  it('gives a project that never changed its workspace an empty view', () => {
    expect(migrate(storedOn('compose'))).toEqual({ structures: {}, views: {}, workspaceId: 'compose' });
  });

  it('keeps what a view can hold, so the saved workspace and the view show what the project showed', () => {
    for (const [name, stored] of Object.entries(viewOnlyFixtures)) {
      const migration = migrate(stored);

      expect(migration.structures, name).toEqual({});
      expect(Object.keys(migration.views).length, name).toBeGreaterThan(0);

      for (const [workspaceId, arrangement] of arrangementsOf(stored, migration.workspaceId)) {
        const layout = resolveLayout(workspaces[workspaceId]!, migration.views[workspaceId], registryAvailability);

        expect(resolvedScreen(layout), `${name}: ${workspaceId}`).toEqual(onScreen(arrangement));

        // Re-running on what it produced changes nothing on screen.
        const again = migrate({
          ...stored,
          floatingWidgets: layout.floatingWidgets,
          layout: { ...stored.layout, panels: layout.panels, presetId: workspaceId },
          presetWorkingLayouts: undefined,
          widgetInstances: layout.widgetInstances,
          widgetRegions: layout.widgetRegions,
        });

        expect(resolveLayout(workspaces[workspaceId]!, again.views[workspaceId], registryAvailability)).toEqual(layout);
      }
    }
  });

  it('returns a structural change as a structure to adopt, and a view of only what still applies to drop it', () => {
    for (const [name, stored] of Object.entries(structuralFixtures)) {
      const { structures, views, workspaceId } = migrate(stored);
      const adopted = structures[workspaceId];

      expect(adopted, name).toBeDefined();
      expect(resolvedScreen(resolveLayout(adopted!, undefined, registryAvailability)), name).toEqual(onScreen(stored));
      expect(pruneView(views[workspaceId], workspaces[workspaceId]!, registryAvailability), name).toBe(
        views[workspaceId]
      );
    }

    expect(migrate(structuralFixtures['floated a widget']).views.compose).toEqual({
      fronts: { right: 'image-map' },
      sizes: { left: 600 },
    });
  });

  it('reads legacy preset ids, and moves a project off a workspace the account no longer has', () => {
    const legacy = viewOnlyFixtures['switched presets, leaving a working copy'];
    const migrated = migrate({
      ...legacy,
      layout: { ...legacy.layout, presetId: 'canvas' },
      presetWorkingLayouts: legacy.presetWorkingLayouts?.map((copy) => ({ ...copy, presetId: 'canvas-default' })),
    });

    expect(migrated).toEqual(migrate(legacy));
    expect(migrated.workspaceId).toBe('edit');

    const onDeleted = {
      ...storedOn('compose', { regions: { left: { sizePx: 600 } } }),
      layout: { ...getLayoutPreset('compose').snapshot.layout, presetId: 'mine' },
    };

    expect(
      migrate({ ...onDeleted, presetWorkingLayouts: [{ ...workingCopyOf('edit', {}), presetId: 'gone' }] })
    ).toEqual({ structures: {}, views: {}, workspaceId: 'compose' });
  });
});
