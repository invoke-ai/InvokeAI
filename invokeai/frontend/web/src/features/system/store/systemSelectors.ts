import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { isEqual, reduce, pickBy } from 'lodash';

export const systemSelector = (state: RootState) => state.system;

export const toastQueueSelector = (state: RootState) => state.system.toastQueue;

export const activeModelSelector = createSelector(
  systemSelector,
  (system) => {
    const { model_list } = system;
    const activeModel = reduce(
      model_list,
      (acc, model, key) => {
        if (model.status === 'active') {
          acc = key;
        }
        return acc;
      },
      ''
    );
    return { ...model_list[activeModel], name: activeModel };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export const diffusersModelsSelector = createSelector(
  systemSelector,
  (system) => {
    const { model_list } = system;

    const diffusersModels = pickBy(model_list, (model, key) => {
      if (model.format === 'diffusers') {
        return { name: key, ...model };
      }
    });

    return diffusersModels;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);
