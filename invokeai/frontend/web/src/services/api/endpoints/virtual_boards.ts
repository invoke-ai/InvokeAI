import queryString from 'query-string';
import type { ImageCategory } from 'services/api/types';

import type { ApiTagDescription } from '..';
import { api, buildV1Url } from '..';

export type VirtualSubBoard = {
  virtual_board_id: string;
  board_name: string;
  date: string;
  image_count: number;
  asset_count: number;
  cover_image_name: string | null;
};

type ImageNamesResult = {
  image_names: string[];
  starred_count: number;
  total_count: number;
};

const buildVirtualBoardsUrl = (path: string = '') => buildV1Url(`virtual_boards/${path}`);

const virtualBoardsApi = api.injectEndpoints({
  endpoints: (build) => ({
    listVirtualBoardsByDate: build.query<VirtualSubBoard[], void>({
      query: () => ({
        url: buildVirtualBoardsUrl('by_date'),
      }),
      providesTags: (): ApiTagDescription[] => ['VirtualBoards', 'FetchOnReconnect'],
    }),

    getVirtualBoardImageNamesByDate: build.query<
      ImageNamesResult,
      {
        date: string;
        starred_first?: boolean;
        order_dir?: 'ASC' | 'DESC';
        categories?: ImageCategory[];
        search_term?: string;
      }
    >({
      query: ({ date, ...params }) => ({
        url: buildVirtualBoardsUrl(
          `by_date/${date}/image_names?${queryString.stringify(params, { arrayFormat: 'none', skipNull: true, skipEmptyString: true })}`
        ),
      }),
      providesTags: (_result, _error, arg): ApiTagDescription[] => [
        { type: 'ImageNameList', id: `virtual_${arg.date}` },
        'FetchOnReconnect',
      ],
    }),
  }),
});

export const { useListVirtualBoardsByDateQuery, useGetVirtualBoardImageNamesByDateQuery } = virtualBoardsApi;
