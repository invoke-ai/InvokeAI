import type { paths } from 'services/api/schema';

import { api, buildV1Url, LIST_TAG } from '..';

const buildImageMovesUrl = (path: string = '') => buildV1Url(`image_moves/${path}`);

type ImageMoveStatusResponse =
  paths['/api/v1/image_moves/status']['get']['responses']['200']['content']['application/json'];

const imageMovesApi = api.injectEndpoints({
  endpoints: (build) => ({
    getImageMoveStatus: build.query<ImageMoveStatusResponse, void>({
      query: () => ({
        url: buildImageMovesUrl('status'),
        method: 'GET',
      }),
      providesTags: ['ImageMoveStatus', 'FetchOnReconnect'],
    }),
    startImageMove: build.mutation<ImageMoveStatusResponse, void>({
      query: () => ({
        url: buildImageMovesUrl('start'),
        method: 'POST',
      }),
      invalidatesTags: ['ImageMoveStatus'],
    }),
    startImageMoveRecovery: build.mutation<ImageMoveStatusResponse, void>({
      query: () => ({
        url: buildImageMovesUrl('recover'),
        method: 'POST',
      }),
      invalidatesTags: [
        'ImageMoveStatus',
        'ImageNameList',
        'ImageCollectionCounts',
        { type: 'ImageCollection', id: LIST_TAG },
      ],
    }),
  }),
});

export const { useGetImageMoveStatusQuery, useStartImageMoveMutation, useStartImageMoveRecoveryMutation } =
  imageMovesApi;
