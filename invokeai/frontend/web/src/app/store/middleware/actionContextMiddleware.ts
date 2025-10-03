import type { Middleware, UnknownAction } from '@reduxjs/toolkit';
import { injectTabActionContext } from 'app/store/util';
import { isCanvasInstanceAction } from 'features/controlLayers/store/canvasSlice';
import { isTabParamsStateAction } from 'features/controlLayers/store/paramsSlice';
import { selectActiveCanvasId } from 'features/controlLayers/store/selectors';
import { selectActiveTab } from 'features/ui/store/uiSelectors';

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
  return isTabParamsStateAction(action) || isCanvasInstanceAction(action);
};
