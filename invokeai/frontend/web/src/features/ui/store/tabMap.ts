export const tabMap = ['txt2img', 'img2img', 'unifiedCanvas', 'nodes', 'modelManager', 'queue'] as const;

export type InvokeTabName = (typeof tabMap)[number];
