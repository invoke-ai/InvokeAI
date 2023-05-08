export const tabMap = [
  'text',
  'image',
  // 'generate',
  'unifiedCanvas',
  'nodes',
  // 'postprocessing',
  // 'training',
] as const;

export type InvokeTabName = (typeof tabMap)[number];
