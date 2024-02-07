export const tabMap = ['txt2img', 'img2img', 'unifiedCanvas', 'workflow', 'nodes', 'modelManager', 'queue'] as const;

export type InvokeTabName = (typeof tabMap)[number];
