export const tabMap = [
  'txt2img',
  'img2img',
  'unifiedCanvas',
  'nodes',
  'modelManager',
  'batch',
] as const;

export type InvokeTabName = (typeof tabMap)[number];
