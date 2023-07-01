import { ModelsList } from 'services/api/types';
import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { keyBy } from 'lodash-es';

import { ApiFullTagDescription, LIST_TAG, api } from '..';
import { paths } from '../schema';

type ModelConfig = ModelsList['models'][number];

type ListModelsArg = NonNullable<
  paths['/api/v1/models/']['get']['parameters']['query']
>;

const modelsAdapter = createEntityAdapter<ModelConfig>({
  selectId: (model) => getModelId(model),
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});

const getModelId = ({ base_model, type, name }: ModelConfig) =>
  `${base_model}/${type}/${name}`;

export const modelsApi = api.injectEndpoints({
  endpoints: (build) => ({
    listModels: build.query<EntityState<ModelConfig>, ListModelsArg>({
      query: (arg) => ({ url: 'models/', params: arg }),
      providesTags: (result, error, arg) => {
        // any list of boards
        const tags: ApiFullTagDescription[] = [{ id: 'Model', type: LIST_TAG }];

        if (result) {
          // and individual tags for each board
          tags.push(
            ...result.ids.map((id) => ({
              type: 'Model' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (response: ModelsList, meta, arg) => {
        return modelsAdapter.setAll(
          modelsAdapter.getInitialState(),
          keyBy(response.models, getModelId)
        );
      },
    }),
  }),
});

export const { useListModelsQuery } = modelsApi;
