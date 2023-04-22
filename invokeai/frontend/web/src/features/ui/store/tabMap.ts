export const tabMap = [
  // 'txt2img',
  // 'img2img',
  'linear',
  'unifiedCanvas',
  'nodes',
  // 'postprocessing',
  // 'training',
] as const;

export type InvokeTabName = (typeof tabMap)[number];
