export const TAB_NUMBER_MAP = ['txt2img', 'img2img', 'unifiedCanvas', 'nodes', 'modelManager', 'queue'] as const;

export type InvokeTabName = (typeof TAB_NUMBER_MAP)[number];
