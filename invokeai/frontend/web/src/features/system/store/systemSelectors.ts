import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { reduce, pickBy } from 'lodash-es';

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
  defaultSelectorOptions
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
  defaultSelectorOptions
);

export const languageSelector = createSelector(
  systemSelector,
  (system) => system.language,
  defaultSelectorOptions
);
