import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { cloneDeep } from 'lodash-es';
import {
  AnyModelConfig,
  ControlNetModelConfig,
  LoRAModelConfig,
  MainModelConfig,
  TextualInversionModelConfig,
  VaeModelConfig,
} from 'services/api/types';

import { ApiFullTagDescription, LIST_TAG, api } from '..';

export type MainModelConfigEntity = MainModelConfig & { id: string };

export type LoRAModelConfigEntity = LoRAModelConfig & { id: string };

export type ControlNetModelConfigEntity = ControlNetModelConfig & {
  id: string;
};

export type TextualInversionModelConfigEntity = TextualInversionModelConfig & {
  id: string;
};

export type VaeModelConfigEntity = VaeModelConfig & { id: string };

type AnyModelConfigEntity =
  | MainModelConfigEntity
  | LoRAModelConfigEntity
  | ControlNetModelConfigEntity
  | TextualInversionModelConfigEntity
  | VaeModelConfigEntity;

const mainModelsAdapter = createEntityAdapter<MainModelConfigEntity>({
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
const loraModelsAdapter = createEntityAdapter<LoRAModelConfigEntity>({
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});
const controlNetModelsAdapter =
  createEntityAdapter<ControlNetModelConfigEntity>({
    sortComparer: (a, b) => a.name.localeCompare(b.name),
  });
const textualInversionModelsAdapter =
  createEntityAdapter<TextualInversionModelConfigEntity>({
    sortComparer: (a, b) => a.name.localeCompare(b.name),
  });
const vaeModelsAdapter = createEntityAdapter<VaeModelConfigEntity>({
  sortComparer: (a, b) => a.name.localeCompare(b.name),
});

export const getModelId = ({ base_model, type, name }: AnyModelConfig) =>
  `${base_model}/${type}/${name}`;

const createModelEntities = <T extends AnyModelConfigEntity>(
  models: AnyModelConfig[]
): T[] => {
  const entityArray: T[] = [];
  models.forEach((model) => {
    const entity = {
      ...cloneDeep(model),
      id: getModelId(model),
    } as T;
    entityArray.push(entity);
  });
  return entityArray;
};

export const modelsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getMainModels: build.query<EntityState<MainModelConfigEntity>, void>({
      query: () => ({ url: 'models/', params: { model_type: 'main' } }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'MainModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'MainModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (
        response: { models: MainModelConfig[] },
        meta,
        arg
      ) => {
        const entities = createModelEntities<MainModelConfigEntity>(
          response.models
        );
        return mainModelsAdapter.setAll(
          mainModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getLoRAModels: build.query<EntityState<LoRAModelConfigEntity>, void>({
      query: () => ({ url: 'models/', params: { model_type: 'lora' } }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'LoRAModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'LoRAModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (
        response: { models: LoRAModelConfig[] },
        meta,
        arg
      ) => {
        const entities = createModelEntities<LoRAModelConfigEntity>(
          response.models
        );
        return loraModelsAdapter.setAll(
          loraModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getControlNetModels: build.query<
      EntityState<ControlNetModelConfigEntity>,
      void
    >({
      query: () => ({ url: 'models/', params: { model_type: 'controlnet' } }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'ControlNetModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'ControlNetModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (
        response: { models: ControlNetModelConfig[] },
        meta,
        arg
      ) => {
        const entities = createModelEntities<ControlNetModelConfigEntity>(
          response.models
        );
        return controlNetModelsAdapter.setAll(
          controlNetModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getVaeModels: build.query<EntityState<VaeModelConfigEntity>, void>({
      query: () => ({ url: 'models/', params: { model_type: 'vae' } }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'VaeModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'VaeModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (
        response: { models: VaeModelConfig[] },
        meta,
        arg
      ) => {
        const entities = createModelEntities<VaeModelConfigEntity>(
          response.models
        );
        return vaeModelsAdapter.setAll(
          vaeModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
    getTextualInversionModels: build.query<
      EntityState<TextualInversionModelConfigEntity>,
      void
    >({
      query: () => ({ url: 'models/', params: { model_type: 'embedding' } }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'TextualInversionModel', id: LIST_TAG },
        ];

        if (result) {
          tags.push(
            ...result.ids.map((id) => ({
              type: 'TextualInversionModel' as const,
              id,
            }))
          );
        }

        return tags;
      },
      transformResponse: (
        response: { models: TextualInversionModelConfig[] },
        meta,
        arg
      ) => {
        const entities = createModelEntities<TextualInversionModelConfigEntity>(
          response.models
        );
        return textualInversionModelsAdapter.setAll(
          textualInversionModelsAdapter.getInitialState(),
          entities
        );
      },
    }),
  }),
});

export const {
  useGetMainModelsQuery,
  useGetControlNetModelsQuery,
  useGetLoRAModelsQuery,
  useGetTextualInversionModelsQuery,
  useGetVaeModelsQuery,
} = modelsApi;
