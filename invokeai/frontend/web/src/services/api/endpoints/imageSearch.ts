import { IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { ListImagesResponse } from 'services/api/types';
import stableHash from 'stable-hash';

import { api, buildV1Url } from '..';

export type StarredMode = 'include' | 'exclude' | 'only';

export type ImageSearchArgs = {
  file_name_enabled: boolean;
  file_name_term: string;
  metadata_enabled: boolean;
  metadata_term: string;
  width_enabled: boolean;
  width_min: string;
  width_max: string;
  width_exact: string;
  height_enabled: boolean;
  height_min: string;
  height_max: string;
  height_exact: string;
  board_ids: string[];
  starred_mode: StarredMode;
  offset: number;
  limit: number;
};

const toInt = (v: string): number | undefined => {
  const trimmed = v.trim();
  if (!trimmed.length) {
    return undefined;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const imageSearchApi = api.injectEndpoints({
  endpoints: (build) => ({
    searchImages: build.query<ListImagesResponse, ImageSearchArgs>({
      query: (arg) => ({
        url: buildV1Url('images/search'),
        method: 'POST',
        params: {
          categories: IMAGE_CATEGORIES,
          is_intermediate: false,
          offset: arg.offset,
          limit: arg.limit,
        },
        body: {
          file_name_term: arg.file_name_enabled ? arg.file_name_term : undefined,
          metadata_term: arg.metadata_enabled ? arg.metadata_term : undefined,
          width_min: arg.width_enabled ? toInt(arg.width_min) : undefined,
          width_max: arg.width_enabled ? toInt(arg.width_max) : undefined,
          width_exact: arg.width_enabled ? toInt(arg.width_exact) : undefined,
          height_min: arg.height_enabled ? toInt(arg.height_min) : undefined,
          height_max: arg.height_enabled ? toInt(arg.height_max) : undefined,
          height_exact: arg.height_enabled ? toInt(arg.height_exact) : undefined,
          board_ids: arg.board_ids.length ? arg.board_ids : undefined,
          starred_mode: arg.starred_mode,
        },
      }),
      providesTags: (result, error, arg) => [
        { type: 'ImageSearchList', id: stableHash(arg) },
        { type: 'ImageSearchList', id: 'LIST' },
        'FetchOnReconnect',
      ],
    }),
  }),
});

export const { useSearchImagesQuery } = imageSearchApi;
