import type {
  BuiltInLayoutPresetId,
  CenterViewId,
  LayoutPreset,
  LayoutPresetRoute,
  LayoutPresetSnapshot,
  LayoutPresetWidgetInstanceSnapshot,
  PanelState,
  WidgetRegion,
  WidgetRegionState,
} from '@workbench/layoutContracts';
import type { WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

import { BUILT_IN_LAYOUT_PRESET_LABELS } from '@workbench/launchpad/intents';

const defaultBottomInstanceIds: WidgetInstanceId[] = [
  'server-status',
  'queue-status',
  'gallery:bottom',
  'notifications',
  'autosave-status',
];

const defaultInstanceTypes: Record<WidgetInstanceId, WidgetTypeId> = {
  'autosave-status': 'autosave-status',
  canvas: 'canvas',
  diagnostics: 'diagnostics',
  'diagnostics:bottom': 'diagnostics',
  gallery: 'gallery',
  'gallery:bottom': 'gallery',
  'gallery:center': 'gallery',
  generate: 'generate',
  'image-map': 'image-map',
  upscale: 'upscale',
  video: 'video',
  layers: 'layers',
  notifications: 'notifications',
  preview: 'preview',
  project: 'project',
  queue: 'queue',
  'queue-status': 'queue-status',
  'server-status': 'server-status',
  'version-status': 'version-status',
  workflow: 'workflow',
  'workflow:bottom': 'workflow',
  'workflow:center': 'workflow',
};

const createRegion = ({
  activeInstanceId,
  instanceIds,
  isCollapsed = false,
  sizePx,
}: {
  activeInstanceId: WidgetInstanceId;
  instanceIds: WidgetInstanceId[];
  isCollapsed?: boolean;
  sizePx: number;
}): WidgetRegionState => ({ activeInstanceId, instanceIds, isCollapsed, sizePx });

const createWidgetInstances = (
  widgetRegions: Record<WidgetRegion, WidgetRegionState>
): Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot> => {
  const widgetInstances: Record<WidgetInstanceId, LayoutPresetWidgetInstanceSnapshot> = {};

  for (const region of Object.values(widgetRegions)) {
    const instanceIds = new Set([region.activeInstanceId, ...region.instanceIds]);

    for (const instanceId of instanceIds) {
      const typeId = defaultInstanceTypes[instanceId];

      if (typeId) {
        widgetInstances[instanceId] = { id: instanceId, typeId };
      }
    }
  }

  return widgetInstances;
};

const createSnapshot = ({
  centerViewId,
  panels,
  presetId,
  widgetRegions,
}: {
  centerViewId: CenterViewId;
  panels: PanelState;
  presetId: BuiltInLayoutPresetId;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
}): LayoutPresetSnapshot => ({
  layout: { centerViewId, panels, presetId },
  widgetInstances: createWidgetInstances(widgetRegions),
  widgetRegions,
});

export interface BuiltInLayoutPresetDescriptor {
  defaultKeys: readonly string[];
  hotkeyId: string;
  preset: BuiltInLayoutPreset;
}

export interface BuiltInLayoutPreset extends LayoutPreset {
  defaultRoute: LayoutPresetRoute;
  id: BuiltInLayoutPresetId;
  isBuiltIn: true;
}

const createPresetDescriptor = ({
  centerViewId,
  defaultKeys,
  defaultRoute,
  hotkeyId,
  id,
  iconId,
  label,
  panels,
  widgetRegions,
}: {
  centerViewId: CenterViewId;
  defaultKeys: readonly string[];
  defaultRoute: LayoutPresetRoute;
  hotkeyId: string;
  id: BuiltInLayoutPresetId;
  iconId: string;
  label: string;
  panels: PanelState;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
}): BuiltInLayoutPresetDescriptor => ({
  defaultKeys,
  hotkeyId,
  preset: {
    defaultRoute,
    iconId,
    id,
    isBuiltIn: true,
    label,
    snapshot: createSnapshot({ centerViewId, panels, presetId: id, widgetRegions }),
  },
});

/**
 * The three shipped presets. Each combines an arrangement with an editable
 * default route. The route is explicit preset data rather than something
 * inferred from whichever graph widgets happen to be placed in the layout.
 */
export const builtInLayoutPresetDescriptors: BuiltInLayoutPresetDescriptor[] = [
  createPresetDescriptor({
    centerViewId: 'preview',
    defaultKeys: ['alt+1'],
    defaultRoute: { destination: 'gallery', sourceId: 'generate' },
    hotkeyId: 'selectComposePreset',
    id: 'compose',
    iconId: 'type',
    label: BUILT_IN_LAYOUT_PRESET_LABELS.compose,
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      // Gallery stays a center view here: the retired `gallery` preset resolves
      // to Compose because it was "Compose with the center view swapped" (see
      // legacyLayoutPresetIds), so the swap has to stay reachable. Curating it
      // out also made Gallery an *addable* widget instead of a placed one,
      // which routed the switch through a menu that warms the Canvas chunk on
      // intent — a chunk a gallery-only session must never download.
      center: createRegion({
        activeInstanceId: 'preview',
        instanceIds: ['preview', 'gallery:center'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'generate',
        instanceIds: ['generate', 'upscale', 'video'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'gallery',
        instanceIds: ['gallery', 'image-map', 'queue'],
        sizePx: 450,
      }),
    },
  }),
  createPresetDescriptor({
    centerViewId: 'canvas',
    defaultKeys: ['alt+2'],
    defaultRoute: { destination: 'canvas', sourceId: 'canvas' },
    hotkeyId: 'selectEditPreset',
    id: 'edit',
    iconId: 'layers',
    label: BUILT_IN_LAYOUT_PRESET_LABELS.edit,
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      center: createRegion({
        activeInstanceId: 'canvas',
        instanceIds: ['canvas', 'preview'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'generate',
        instanceIds: ['generate', 'upscale', 'video'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'layers',
        instanceIds: ['layers', 'preview', 'gallery', 'image-map', 'queue'],
        sizePx: 450,
      }),
    },
  }),
  createPresetDescriptor({
    centerViewId: 'workflow',
    defaultKeys: ['alt+3'],
    defaultRoute: { destination: 'gallery', sourceId: 'workflow' },
    hotkeyId: 'selectAutomatePreset',
    id: 'automate',
    iconId: 'workflow',
    label: BUILT_IN_LAYOUT_PRESET_LABELS.automate,
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      center: createRegion({
        activeInstanceId: 'workflow:center',
        instanceIds: ['workflow:center', 'preview'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'workflow',
        instanceIds: ['workflow'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'queue',
        instanceIds: ['queue', 'preview', 'gallery', 'image-map'],
        sizePx: 450,
      }),
    },
  }),
];

export const layoutPresets: BuiltInLayoutPreset[] = builtInLayoutPresetDescriptors.map(({ preset }) => preset);

export const defaultLayoutPreset = layoutPresets[0]!;

/**
 * Preset ids persisted before the three-preset model. `gallery` had no successor
 * arrangement of its own — it was Compose with the center view swapped — so it
 * resolves there rather than becoming a fourth entry.
 */
const legacyLayoutPresetIds: Record<string, BuiltInLayoutPresetId> = {
  canvas: 'edit',
  'canvas-default': 'compose',
  gallery: 'compose',
  workflow: 'automate',
};

/**
 * Rewrites a persisted preset id onto the current set. Custom preset ids pass
 * through untouched — they are resolved against the account's own list, and only
 * the built-in ids were ever renamed.
 */
export const resolveLayoutPresetId = (presetId: string): string => legacyLayoutPresetIds[presetId] ?? presetId;

export const getLayoutPreset = (presetId: string): BuiltInLayoutPreset =>
  layoutPresets.find((preset) => preset.id === resolveLayoutPresetId(presetId)) ?? defaultLayoutPreset;

export const isBuiltInLayoutPresetId = (presetId: string): presetId is BuiltInLayoutPresetId =>
  layoutPresets.some((preset) => preset.id === presetId);
