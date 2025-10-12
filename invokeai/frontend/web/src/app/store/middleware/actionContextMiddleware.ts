import type { Middleware, UnknownAction } from '@reduxjs/toolkit';
import { injectTabActionContext } from 'app/store/util';
import { isCanvasInstanceAction } from 'features/controlLayers/store/canvasSlice';
import { selectActiveCanvasId, selectActiveTab } from 'features/controlLayers/store/selectors';
import { isTabInstanceParamsAction } from 'features/controlLayers/store/tabSlice';

export const actionContextMiddleware: Middleware = (store) => (next) => (action) => {
  const currentAction = action as UnknownAction;

  if (isTabActionContextRequired(currentAction)) {
    const state = store.getState();
    const tab = selectActiveTab(state);
    const canvasId = tab === 'canvas' ? selectActiveCanvasId(state) : undefined;

    injectTabActionContext(currentAction, tab, canvasId);
  }

  return next(action);
};

const isTabActionContextRequired = (action: UnknownAction) => {
  return isTabInstanceParamsAction(action) || isCanvasInstanceAction(action);
};
