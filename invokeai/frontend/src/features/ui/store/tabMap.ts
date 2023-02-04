export const tabMap = [
  'txt2img',
  'img2img',
  'unifiedCanvas',
  'nodes',
  'postprocess',
  'training',
] as const;

export type InvokeTabName = (typeof tabMap)[number];
