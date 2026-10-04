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

// Notifications read as ambient status, not a tool: they live on the right.
const defaultBottomAlignEndInstanceIds: WidgetInstanceId[] = ['notifications'];
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
  workflow: 'workflow',
  'workflow:bottom': 'workflow',
  'workflow:center': 'workflow',
};

const createRegion = ({
  activeInstanceId,
  alignEndInstanceIds,
  instanceIds,
  isCollapsed = false,
  sizePx,
}: {
  activeInstanceId: WidgetInstanceId;
  alignEndInstanceIds?: WidgetInstanceId[];
  instanceIds: WidgetInstanceId[];
  isCollapsed?: boolean;
  sizePx: number;
}): WidgetRegionState => ({
  activeInstanceId,
  ...(alignEndInstanceIds ? { alignEndInstanceIds } : {}),
  instanceIds,
  isCollapsed,
  sizePx,
});

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

/** Presets define arrangement and default route explicitly; placed graph widgets do not determine the route. */
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
        alignEndInstanceIds: defaultBottomAlignEndInstanceIds,
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      // Keep Gallery placed in Compose so migrated gallery presets remain reachable without the add-widget path
      // warming Canvas.
      center: createRegion({
        activeInstanceId: 'preview',
        instanceIds: ['preview', 'gallery:center'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'generate',
        instanceIds: ['generate', 'upscale'],
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
        alignEndInstanceIds: defaultBottomAlignEndInstanceIds,
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
        instanceIds: ['generate', 'upscale'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'layers',
        // Place Preview behind Layers so its float/dock and floated Invoke controls remain reachable.
        instanceIds: ['layers', 'preview'],
        sizePx: 450,
      }),
    },
  }),
  createPresetDescriptor({
    centerViewId: 'preview',
    defaultKeys: ['alt+3'],
    defaultRoute: { destination: 'gallery', sourceId: 'video' },
    hotkeyId: 'selectVideoPreset',
    id: 'video',
    iconId: 'clapperboard',
    label: BUILT_IN_LAYOUT_PRESET_LABELS.video,
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        alignEndInstanceIds: defaultBottomAlignEndInstanceIds,
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      center: createRegion({
        activeInstanceId: 'preview',
        instanceIds: ['preview'],
        sizePx: 0,
      }),
      // Video leads the rail; Upscale stays behind it for finishing a still.
      left: createRegion({
        activeInstanceId: 'video',
        instanceIds: ['video', 'upscale'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'gallery',
        instanceIds: ['gallery', 'queue'],
        sizePx: 450,
      }),
    },
  }),
  createPresetDescriptor({
    centerViewId: 'workflow',
    defaultKeys: ['alt+4'],
    defaultRoute: { destination: 'gallery', sourceId: 'workflow' },
    hotkeyId: 'selectAutomatePreset',
    id: 'automate',
    iconId: 'workflow',
    label: BUILT_IN_LAYOUT_PRESET_LABELS.automate,
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        alignEndInstanceIds: defaultBottomAlignEndInstanceIds,
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

/** The historical gallery preset was a Compose center-view variant, so it maps to Compose. */
const legacyLayoutPresetIds: Record<string, BuiltInLayoutPresetId> = {
  canvas: 'edit',
  'canvas-default': 'compose',
  gallery: 'compose',
  workflow: 'automate',
};

/** Remap historical built-in ids; account-defined custom ids pass through unchanged. */
export const resolveLayoutPresetId = (presetId: string): string => legacyLayoutPresetIds[presetId] ?? presetId;

export const getLayoutPreset = (presetId: string): BuiltInLayoutPreset =>
  layoutPresets.find((preset) => preset.id === resolveLayoutPresetId(presetId)) ?? defaultLayoutPreset;

export const isBuiltInLayoutPresetId = (presetId: string): presetId is BuiltInLayoutPresetId =>
  layoutPresets.some((preset) => preset.id === presetId);
