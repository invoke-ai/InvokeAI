import type { UnknownAction } from '@reduxjs/toolkit';
import type { TabName } from 'features/controlLayers/store/types';

const TAB_KEY = Symbol('tab');
const CANVAS_ID_KEY = Symbol('canvasId');

type TabActionContext = {
  [TAB_KEY]: TabName;
  [CANVAS_ID_KEY]?: string;
};

export const injectTabActionContext = (action: UnknownAction, tab: TabName, canvasId?: string) => {
  const context: TabActionContext = canvasId ? { [TAB_KEY]: tab, [CANVAS_ID_KEY]: canvasId } : { [TAB_KEY]: tab };
  Object.assign(action, { meta: context });
};

export const extractTabActionContext = (action: UnknownAction & { meta?: Partial<TabActionContext> }) => {
  const tab = action.meta?.[TAB_KEY];
  const canvasId = action.meta?.[CANVAS_ID_KEY];

  if (!tab || (tab === 'canvas' && !canvasId)) {
    return undefined;
  }

  return {
    tab,
    canvasId,
  };
};
