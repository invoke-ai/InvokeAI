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

// Not exported: with virtual-date name lists served by `listGalleryItemNames`, nothing outside
// this module needs the api object itself — only the hook below.
const virtualBoardsApi = api.injectEndpoints({
  endpoints: (build) => ({
    listVirtualBoardsByDate: build.query<VirtualSubBoard[], void>({
      query: () => ({
        url: buildVirtualBoardsUrl('by_date'),
      }),
      providesTags: (): ApiTagDescription[] => ['VirtualBoards', 'FetchOnReconnect'],
    }),
  }),
});

// Virtual-date name lists are served by `listGalleryItemNames` with a `created_date` filter;
// the deprecated `by_date/{date}/item_names` route is no longer called from the UI.
export const { useListVirtualBoardsByDateQuery } = virtualBoardsApi;
