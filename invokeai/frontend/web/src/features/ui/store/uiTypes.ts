import { isPlainObject } from 'es-toolkit';
import { z } from 'zod';

export const zTabName = z.enum(['generate', 'canvas', 'upscaling', 'workflows', 'models', 'queue', 'video']);
export type TabName = z.infer<typeof zTabName>;

const zPartialDimensions = z.object({
  width: z.number().optional(),
  height: z.number().optional(),
});

const zSerializable = z.any().refine(isPlainObject);
export type Serializable = z.infer<typeof zSerializable>;

export const zUIState = z.object({
  _version: z.literal(4),
  activeTab: zTabName,
  shouldShowImageDetails: z.boolean(),
  shouldShowProgressInViewer: z.boolean(),
  accordions: z.record(z.string(), z.boolean()),
  expanders: z.record(z.string(), z.boolean()),
  textAreaSizes: z.record(z.string(), zPartialDimensions),
  panels: z.record(z.string(), zSerializable),
  shouldShowNotificationV2: z.boolean(),
  pickerCompactViewStates: z.record(z.string(), z.boolean()),
});
export type UIState = z.infer<typeof zUIState>;
export const getInitialUIState = (): UIState => ({
  _version: 4 as const,
  activeTab: 'generate' as const,
  shouldShowImageDetails: false,
  shouldShowProgressInViewer: true,
  accordions: {},
  expanders: {},
  textAreaSizes: {},
  panels: {},
  shouldShowNotificationV2: true,
  pickerCompactViewStates: {},
});
