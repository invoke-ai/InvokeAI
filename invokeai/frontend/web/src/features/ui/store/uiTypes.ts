import { deepClone } from 'common/util/deepClone';
import { isPlainObject } from 'es-toolkit';
import { z } from 'zod';

const zTabName = z.enum(['generate', 'canvas', 'upscaling', 'workflows', 'models', 'queue']);
export type TabName = z.infer<typeof zTabName>;

const zPartialDimensions = z.object({
  width: z.number().optional(),
  height: z.number().optional(),
});

const zSerializable = z.any().refine(isPlainObject);
export type Serializable = z.infer<typeof zSerializable>;

const zUIState = z.object({
  _version: z.literal(3).default(3),
  activeTab: zTabName.default('generate'),
  shouldShowImageDetails: z.boolean().default(false),
  shouldShowProgressInViewer: z.boolean().default(true),
  accordions: z.record(z.string(), z.boolean()).default(() => ({})),
  expanders: z.record(z.string(), z.boolean()).default(() => ({})),
  textAreaSizes: z.record(z.string(), zPartialDimensions).default({}),
  panels: z.record(z.string(), zSerializable).default({}),
  shouldShowNotificationV2: z.boolean().default(true),
  pickerCompactViewStates: z.record(z.string(), z.boolean()).default(() => ({})),
});
const INITIAL_STATE = zUIState.parse({});
export type UIState = z.infer<typeof zUIState>;
export const getInitialUIState = (): UIState => deepClone(INITIAL_STATE);
