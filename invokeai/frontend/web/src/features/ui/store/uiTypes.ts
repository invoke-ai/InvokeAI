import { deepClone } from 'common/util/deepClone';
import { z } from 'zod/v4';

const zTabName = z.enum(['generate', 'canvas', 'upscaling', 'workflows', 'models', 'queue']);
export type TabName = z.infer<typeof zTabName>;
const zCanvasRightPanelTabName = z.enum(['layers', 'gallery']);
const zCanvasMainPanelTabName = z.enum(['launchpad', 'workspace', 'viewer']);
const zGenerateMainPanelTabName = z.enum(['launchpad', 'viewer']);
const zUpscalingMainPanelTabName = z.enum(['launchpad', 'viewer']);
const zWorkflowsMainPanelTabName = z.enum(['launchpad', 'workspace', 'viewer']);

const zPartialDimensions = z.object({
  width: z.number().optional(),
  height: z.number().optional(),
});

// Panel state types for Gridview and Dockview panels
const zGridviewPanelState = z.object({
  width: z.number().optional(),
  height: z.number().optional(),
});

const zDockviewPanelState = z.object({
  isActive: z.boolean().optional(),
});

const zUIState = z.object({
  _version: z.literal(4).default(4),
  activeTab: zTabName.default('canvas'),
  activeTabCanvasRightPanel: zCanvasRightPanelTabName.default('gallery'),
  activeTabCanvasMainPanel: zCanvasMainPanelTabName.default('launchpad'),
  activeTabGenerateMainPanel: zGenerateMainPanelTabName.default('launchpad'),
  activeTabUpscalingMainPanel: zUpscalingMainPanelTabName.default('launchpad'),
  activeTabWorkflowsMainPanel: zWorkflowsMainPanelTabName.default('launchpad'),
  shouldShowImageDetails: z.boolean().default(false),
  shouldShowProgressInViewer: z.boolean().default(true),
  accordions: z.record(z.string(), z.boolean()).default(() => ({})),
  expanders: z.record(z.string(), z.boolean()).default(() => ({})),
  textAreaSizes: z.record(z.string(), zPartialDimensions).default({}),
  gridviewPanelStates: z.record(z.string(), zGridviewPanelState).default({}),
  dockviewPanelStates: z.record(z.string(), zDockviewPanelState).default({}),
  shouldShowNotificationV2: z.boolean().default(true),
});
const INITIAL_STATE = zUIState.parse({});
export type UIState = z.infer<typeof zUIState>;
export type GridviewPanelState = z.infer<typeof zGridviewPanelState>;
export type DockviewPanelState = z.infer<typeof zDockviewPanelState>;
export type CanvasMainPanelTabName = z.infer<typeof zCanvasMainPanelTabName>;
export type GenerateMainPanelTabName = z.infer<typeof zGenerateMainPanelTabName>;
export type UpscalingMainPanelTabName = z.infer<typeof zUpscalingMainPanelTabName>;
export type WorkflowsMainPanelTabName = z.infer<typeof zWorkflowsMainPanelTabName>;
export const getInitialUIState = (): UIState => deepClone(INITIAL_STATE);
