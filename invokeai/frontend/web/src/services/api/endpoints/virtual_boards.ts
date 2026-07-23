import queryString from 'query-string';
import type { GetGalleryItemNamesResult, ImageCategory } from 'services/api/types';

import type { ApiTagDescription } from '..';
import { api, buildV1Url } from '..';

export type VirtualSubBoard = {
  virtual_board_id: string;
  board_name: string;
  date: string;
  image_count: number;
  asset_count: number;
  video_count: number;
  cover_image_name: string | null;
  cover_video_name: string | null;
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

    /**
     * Polymorphic (image + video) refs for a virtual date board. Same result shape as the
     * gallery's `getGalleryItemNames`, so the gallery grid can consume either transparently.
     */
    getVirtualBoardItemNamesByDate: build.query<
      GetGalleryItemNamesResult,
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
          `by_date/${date}/item_names?${queryString.stringify(params, { arrayFormat: 'none', skipNull: true, skipEmptyString: true })}`
        ),
      }),
      // Both image and video mutations must refetch a virtual date's contents, so this
      // provides the name-list tag of each kind plus the polymorphic one.
      providesTags: (_result, _error, arg): ApiTagDescription[] => [
        { type: 'ImageNameList', id: `virtual_${arg.date}` },
        { type: 'VideoNameList', id: `virtual_${arg.date}` },
        'GalleryItemNameList',
        'FetchOnReconnect',
      ],
    }),
  }),
});

export const { useListVirtualBoardsByDateQuery, useGetVirtualBoardItemNamesByDateQuery } = virtualBoardsApi;
