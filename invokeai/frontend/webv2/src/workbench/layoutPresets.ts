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

const defaultBottomInstanceIds: WidgetInstanceId[] = [
  'server-status',
  'diagnostics:bottom',
  'gallery:bottom',
  'notifications',
  'autosave-status',
  'version-status',
  'workflow:bottom',
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
  upscale: 'upscale',
  layers: 'layers',
  notifications: 'notifications',
  preview: 'preview',
  project: 'project',
  queue: 'queue',
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
  iconId: string;
  preset: LayoutPreset;
  tooltip: string;
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
  tooltip,
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
  tooltip: string;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
}): BuiltInLayoutPresetDescriptor => ({
  defaultKeys,
  hotkeyId,
  iconId,
  preset: {
    defaultRoute,
    iconId,
    id,
    isBuiltIn: true,
    label,
    snapshot: createSnapshot({ centerViewId, panels, presetId: id, widgetRegions }),
  },
  tooltip,
});

/**
 * The three shipped presets. Each is an arrangement, named for the work it
 * supports rather than the widget it happens to open — several graph widgets
 * are placed in every one of them, which is exactly why a preset can never
 * imply an invocation source.
 */
export const builtInLayoutPresetDescriptors: BuiltInLayoutPresetDescriptor[] = [
  createPresetDescriptor({
    centerViewId: 'preview',
    defaultKeys: ['alt+1'],
    defaultRoute: { destination: 'gallery', sourceId: 'generate' },
    hotkeyId: 'selectComposePreset',
    id: 'compose',
    iconId: 'type',
    label: 'Compose',
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    tooltip: 'Text to image',
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      center: createRegion({
        activeInstanceId: 'preview',
        instanceIds: ['preview', 'canvas', 'gallery:center', 'workflow:center'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'generate',
        instanceIds: ['generate', 'workflow', 'upscale'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'gallery',
        instanceIds: ['gallery', 'preview', 'queue', 'layers', 'diagnostics', 'project'],
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
    label: 'Edit',
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    tooltip: 'Canvas editing',
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      center: createRegion({
        activeInstanceId: 'canvas',
        instanceIds: ['canvas', 'preview', 'gallery:center', 'workflow:center'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'generate',
        instanceIds: ['generate', 'workflow', 'upscale'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'layers',
        instanceIds: ['layers', 'gallery', 'queue', 'preview', 'diagnostics', 'project'],
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
    label: 'Automate',
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    tooltip: 'Node workflows',
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'workflow:bottom',
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      center: createRegion({
        activeInstanceId: 'workflow:center',
        instanceIds: ['workflow:center', 'canvas', 'preview', 'gallery:center'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'workflow',
        instanceIds: ['workflow', 'generate', 'upscale'],
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'queue',
        instanceIds: ['queue', 'gallery', 'layers', 'preview', 'diagnostics', 'project'],
        sizePx: 450,
      }),
    },
  }),
];

export const layoutPresets: LayoutPreset[] = builtInLayoutPresetDescriptors.map(({ preset }) => preset);

export const defaultLayoutPreset = layoutPresets[0];

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

export const getLayoutPreset = (presetId: string) =>
  layoutPresets.find((preset) => preset.id === resolveLayoutPresetId(presetId)) ?? defaultLayoutPreset;

export const isBuiltInLayoutPresetId = (presetId: string): presetId is BuiltInLayoutPresetId =>
  layoutPresets.some((preset) => preset.id === presetId);
