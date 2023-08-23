import { UIState } from './uiTypes';

/**
 * UI slice persist denylist
 */
export const uiPersistDenylist: (keyof UIState)[] = [
  'shouldShowImageDetails',
  'globalContextMenuCloseTrigger',
  'panels',
];
