export const tabMap = [
  'txt2img',
  'img2img',
  'unifiedCanvas',
  'nodes',
  'postprocessing',
  'training',
] as const;

export type InvokeTabName = (typeof tabMap)[number];
