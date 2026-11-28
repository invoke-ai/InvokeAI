/** Test-only: layouts as projects stored them before structure and view split, written as those builds wrote them. */
import type { LayoutPresetSnapshot, WidgetRegion, WidgetRegionState } from '@workbench/layoutContracts';

import { getLayoutPreset } from '@workbench/layoutPresets';

import type { StoredProjectLayout } from './migrateProjectLayout';

const withRegions = (
  snapshot: LayoutPresetSnapshot,
  patches: Partial<Record<WidgetRegion, Partial<WidgetRegionState>>>
): LayoutPresetSnapshot['widgetRegions'] =>
  Object.fromEntries(
    Object.entries(snapshot.widgetRegions).map(([region, state]) => [
      region,
      { ...state, ...patches[region as WidgetRegion] },
    ])
  ) as LayoutPresetSnapshot['widgetRegions'];

/** A project on a built-in that stored its arrangement with `regions` (and windows) changed from the shipped one. */
export const storedOn = (
  presetId: 'automate' | 'compose' | 'edit' | 'video',
  {
    floatingWidgets,
    regions = {},
    workingLayouts,
  }: Partial<Pick<StoredProjectLayout, 'floatingWidgets'>> & {
    regions?: Partial<Record<WidgetRegion, Partial<WidgetRegionState>>>;
    workingLayouts?: StoredProjectLayout['presetWorkingLayouts'];
  } = {}
): StoredProjectLayout => {
  const { snapshot } = getLayoutPreset(presetId);

  return {
    ...(floatingWidgets ? { floatingWidgets } : {}),
    layout: { ...snapshot.layout },
    ...(workingLayouts ? { presetWorkingLayouts: workingLayouts } : {}),
    widgetInstances: snapshot.widgetInstances,
    widgetRegions: withRegions(snapshot, regions),
  };
};

/** A working copy of a built-in, as a project kept one for a preset it was not on. */
export const workingCopyOf = (
  presetId: 'automate' | 'compose' | 'edit' | 'video',
  regions: Partial<Record<WidgetRegion, Partial<WidgetRegionState>>>
): NonNullable<StoredProjectLayout['presetWorkingLayouts']>[number] => {
  const { snapshot } = getLayoutPreset(presetId);

  return { presetId, snapshot: { ...snapshot, widgetRegions: withRegions(snapshot, regions) } };
};

/** Arrangements a view can hold: fronts, sizes, collapsed panels and a working copy of another preset. */
export const viewOnlyLayouts = {
  'fronted other tabs': storedOn('compose', {
    regions: { center: { activeInstanceId: 'gallery:center' }, right: { activeInstanceId: 'queue' } },
  }),
  'resized and collapsed': storedOn('compose', { regions: { left: { sizePx: 600 }, right: { isCollapsed: true } } }),
  'switched presets, leaving a working copy': {
    ...storedOn('edit', { regions: { center: { activeInstanceId: 'preview' } } }),
    presetWorkingLayouts: [workingCopyOf('compose', { right: { sizePx: 401 } })],
  },
} satisfies Record<string, StoredProjectLayout>;

/** Arrangements that place widgets differently from their workspace. */
export const structuralLayouts = {
  'floated a widget': storedOn('compose', {
    floatingWidgets: {
      gallery: {
        heightPx: 320,
        mode: 'windowed',
        returnIndex: 0,
        returnRegion: 'right',
        stackOrder: 1,
        widthPx: 440,
        x: 210,
        y: 140,
      },
    },
    regions: { left: { sizePx: 600 }, right: { activeInstanceId: 'image-map', instanceIds: ['image-map', 'queue'] } },
  }),
  'moved a tab to another rail': storedOn('compose', {
    regions: {
      left: { activeInstanceId: 'image-map', instanceIds: ['generate', 'image-map', 'upscale'] },
      right: { instanceIds: ['gallery', 'queue'] },
    },
  }),
  'removed a widget': storedOn('compose', { regions: { right: { instanceIds: ['gallery', 'queue'] } } }),
} satisfies Record<string, StoredProjectLayout>;
