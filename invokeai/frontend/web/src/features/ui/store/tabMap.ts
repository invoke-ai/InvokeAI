export const tabMap = [
  'txt2img',
  'img2img',
  // 'generate',
  'unifiedCanvas',
  'nodes',
  'batch',
  // 'postprocessing',
  // 'training',
  'modelmanager',
] as const;

export type InvokeTabName = (typeof tabMap)[number];
