import type {
  BuiltInLayoutPresetId,
  CenterViewId,
  LayoutPreset,
  LayoutPresetSnapshot,
  LayoutPresetWidgetInstanceSnapshot,
  PanelState,
  WidgetInstanceId,
  WidgetRegion,
  WidgetRegionState,
  WidgetTypeId,
} from './types';

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

const createPreset = ({
  centerViewId,
  id,
  label,
  panels,
  widgetRegions,
}: {
  centerViewId: CenterViewId;
  id: BuiltInLayoutPresetId;
  label: string;
  panels: PanelState;
  widgetRegions: Record<WidgetRegion, WidgetRegionState>;
}): LayoutPreset => ({
  id,
  isBuiltIn: true,
  label,
  snapshot: createSnapshot({ centerViewId, panels, presetId: id, widgetRegions }),
});

export const layoutPresets: LayoutPreset[] = [
  createPreset({
    centerViewId: 'preview',
    id: 'canvas-default',
    label: 'Default',
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
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
  createPreset({
    centerViewId: 'workflow',
    id: 'workflow',
    label: 'Workflow',
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
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
  createPreset({
    centerViewId: 'canvas',
    id: 'canvas',
    label: 'Canvas',
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
  createPreset({
    centerViewId: 'gallery',
    id: 'gallery',
    label: 'Gallery',
    panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: true },
    widgetRegions: {
      bottom: createRegion({
        activeInstanceId: 'gallery:bottom',
        instanceIds: defaultBottomInstanceIds,
        isCollapsed: true,
        sizePx: 180,
      }),
      center: createRegion({
        activeInstanceId: 'gallery:center',
        instanceIds: ['gallery:center', 'preview', 'canvas', 'workflow:center'],
        sizePx: 0,
      }),
      left: createRegion({
        activeInstanceId: 'generate',
        instanceIds: ['generate', 'workflow', 'upscale', 'gallery'],
        isCollapsed: true,
        sizePx: 450,
      }),
      right: createRegion({
        activeInstanceId: 'preview',
        instanceIds: ['preview', 'gallery', 'queue', 'layers', 'diagnostics', 'project'],
        sizePx: 450,
      }),
    },
  }),
];

export const defaultLayoutPreset = layoutPresets[0];

export const getLayoutPreset = (presetId: string) =>
  layoutPresets.find((preset) => preset.id === presetId) ?? defaultLayoutPreset;
