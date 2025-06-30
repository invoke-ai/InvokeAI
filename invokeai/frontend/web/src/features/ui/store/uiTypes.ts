import { deepClone } from 'common/util/deepClone';
import { z } from 'zod/v4';

const zTabName = z.enum(['generate', 'canvas', 'upscaling', 'workflows', 'models', 'queue']);
export const ALL_TABS = zTabName.options;
export type TabName = z.infer<typeof zTabName>;
const zCanvasRightPanelTabName = z.enum(['layers', 'gallery']);
export type CanvasRightPanelTabName = z.infer<typeof zCanvasRightPanelTabName>;

const zPartialDimensions = z.object({
  width: z.number().optional(),
  height: z.number().optional(),
});
export type PartialDimensions = z.infer<typeof zPartialDimensions>;

export const zUIState = z.object({
  _version: z.literal(3).default(3),
  activeTab: zTabName.default('canvas'),
  activeTabCanvasRightPanel: zCanvasRightPanelTabName.default('gallery'),
  shouldShowImageDetails: z.boolean().default(false),
  shouldShowProgressInViewer: z.boolean().default(true),
  accordions: z.record(z.string(), z.boolean()).default(() => ({})),
  expanders: z.record(z.string(), z.boolean()).default(() => ({})),
  textAreaSizes: z.record(z.string(), zPartialDimensions).default({}),
  shouldShowNotificationV2: z.boolean().default(true),
  showGenerateTabSplashScreen: z.boolean().default(true),
  showCanvasTabSplashScreen: z.boolean().default(true),
});
const INITIAL_STATE = zUIState.parse({});
export type UIState = z.infer<typeof zUIState>;
export const getInitialUIState = (): UIState => deepClone(INITIAL_STATE);
