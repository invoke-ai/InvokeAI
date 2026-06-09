import type { LayoutPreset } from './types';

export const layoutPresets: LayoutPreset[] = [
  {
    description: 'Generate controls, canvas, preview, and workflow tabs in one project-owned workbench.',
    id: 'canvas-default',
    initialLayout: {
      centerViewId: 'canvas',
      panels: { isBottomOpen: true, isLeftOpen: true, isRightOpen: true },
      presetId: 'canvas-default',
    },
    label: 'Default Layout',
  },
  {
    description: 'Future gallery-first page parity preset.',
    id: 'gallery',
    initialLayout: {
      centerViewId: 'gallery',
      panels: { isBottomOpen: true, isLeftOpen: false, isRightOpen: true },
      presetId: 'gallery',
    },
    label: 'Gallery',
  },
  {
    description: 'Future graph authoring preset.',
    id: 'workflow',
    initialLayout: {
      centerViewId: 'workflow',
      panels: { isBottomOpen: false, isLeftOpen: true, isRightOpen: false },
      presetId: 'workflow',
    },
    label: 'Workflow',
  },
  {
    description: 'Future linear generation preset.',
    id: 'linear',
    initialLayout: {
      centerViewId: 'canvas',
      panels: { isBottomOpen: true, isLeftOpen: true, isRightOpen: false },
      presetId: 'linear',
    },
    label: 'Linear UI',
  },
  {
    description: 'Future model management preset.',
    id: 'model-manager',
    initialLayout: {
      centerViewId: 'gallery',
      panels: { isBottomOpen: false, isLeftOpen: false, isRightOpen: true },
      presetId: 'model-manager',
    },
    label: 'Model Manager',
  },
];

export const defaultLayoutPreset = layoutPresets[0];

export const getLayoutPreset = (presetId: string) =>
  layoutPresets.find((preset) => preset.id === presetId) ?? defaultLayoutPreset;
