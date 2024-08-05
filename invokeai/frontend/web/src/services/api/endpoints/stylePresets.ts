import type { paths } from 'services/api/schema';

import { api, buildV1Url, LIST_TAG } from '..';

export type StylePresetRecordDTO = paths['/api/v1/style_presets/i/{style_preset_id}']['get']['responses']['200']['content']['application/json']

/**
 * Builds an endpoint URL for the style_presets router
 * @example
 * buildStylePresetsUrl('some-path')
 * // '/api/v1/style_presets/some-path'
 */
const buildStylePresetsUrl = (path: string = '') => buildV1Url(`style_presets/${path}`);

export const stylePresetsApi = api.injectEndpoints({
  endpoints: (build) => ({
    getStylePreset: build.query<
      paths['/api/v1/style_presets/i/{style_preset_id}']['get']['responses']['200']['content']['application/json'],
      string
    >({
      query: (style_preset_id) => buildStylePresetsUrl(`i/${style_preset_id}`),
      providesTags: (result, error, style_preset_id) => [
        { type: 'StylePreset', id: style_preset_id },
        'FetchOnReconnect',
      ],
    }),
    deleteStylePreset: build.mutation<void, string>({
      query: (style_preset_id) => ({
        url: buildStylePresetsUrl(`i/${style_preset_id}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, style_preset_id) => [
        { type: 'StylePreset', id: LIST_TAG },
        { type: 'StylePreset', id: style_preset_id },
      ],
    }),
    createStylePreset: build.mutation<
      paths['/api/v1/style_presets/']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/style_presets/']['post']['requestBody']['content']['application/json']['style_preset']
    >({
      query: (style_preset) => ({
        url: buildStylePresetsUrl(),
        method: 'POST',
        body: { style_preset },
      }),
      invalidatesTags: [
        { type: 'StylePreset', id: LIST_TAG },
        { type: 'StylePreset', id: LIST_TAG },
      ],
    }),
    updateStylePreset: build.mutation<
      paths['/api/v1/style_presets/i/{style_preset_id}']['patch']['responses']['200']['content']['application/json'],
      {
        id: string;
        changes: paths['/api/v1/style_presets/i/{style_preset_id}']['patch']['requestBody']['content']['application/json']['changes'];
      }
    >({
      query: ({ id, changes }) => ({
        url: buildStylePresetsUrl(`i/${id}`),
        method: 'PATCH',
        body: { changes },
      }),
      invalidatesTags: (response, error, { id }) => [
        { type: 'StylePreset', id: LIST_TAG },
        { type: 'StylePreset', id: id },
      ],
    }),
    listStylePresets: build.query<
      paths['/api/v1/style_presets/']['get']['responses']['200']['content']['application/json'],
      NonNullable<paths['/api/v1/style_presets/']['get']['parameters']['query']>
    >({
      query: (params) => ({
        url: buildStylePresetsUrl(),
        params,
      }),
      providesTags: ['FetchOnReconnect', { type: 'StylePreset', id: LIST_TAG }],
    }),
  }),
});

export const {
  useGetStylePresetQuery,
  useCreateStylePresetMutation,
  useDeleteStylePresetMutation,
  useUpdateStylePresetMutation,
  useListStylePresetsQuery,
} = stylePresetsApi;
