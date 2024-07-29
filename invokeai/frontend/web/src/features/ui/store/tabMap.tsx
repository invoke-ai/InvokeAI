export const TAB_NUMBER_MAP = ['generation', 'canvas', 'upscaling', 'workflows', 'models', 'queue'] as const;

export type InvokeTabName = (typeof TAB_NUMBER_MAP)[number];
