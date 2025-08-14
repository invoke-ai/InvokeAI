import type { paths } from 'services/api/schema';

import { api, buildV1Url, LIST_TAG } from '..';
import { getTagsToInvalidateForBoardAffectingMutation,getTagsToInvalidateForImageMutation } from './images';
import { getTagsToInvalidateForVideoMutation } from './videos';

/**
 * Builds an endpoint URL for the resources router
 * @example
 * buildResourcesUrl('some-path')
 * // '/api/v1/resources/some-path'
 */
const buildResourcesUrl = (path: string = '') => buildV1Url(`resources/${path}`);

/**
 * Builds an endpoint URL for the board_resources router
 * @example
 * buildBoardResourcesUrl('some-path')
 * // '/api/v1/board_resources/some-path'
 */
const buildBoardResourcesUrl = (path: string = '') => buildV1Url(`board_resources/${path}`);

export const resourcesApi = api.injectEndpoints({
  endpoints: (build) => ({
    addResourcesToBoard: build.mutation<
      paths['/api/v1/board_resources/batch']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_resources/batch']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildBoardResourcesUrl('batch'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        const image_ids = result.added_resources.filter(r => r.resource_type === "image").map(r => r.resource_id);
        const video_ids = result.added_resources.filter(r => r.resource_type === "video").map(r => r.resource_id);
        return [
          ...getTagsToInvalidateForImageMutation(image_ids),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          ...getTagsToInvalidateForVideoMutation(video_ids),
        ];
      },
    }),
    removeResourcesFromBoard: build.mutation<
      paths['/api/v1/board_resources/batch/delete']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_resources/batch/delete']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildBoardResourcesUrl('batch/delete'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        const image_ids = result.removed_resources.filter(r => r.resource_type === "image").map(r => r.resource_id);
        const video_ids = result.removed_resources.filter(r => r.resource_type === "video").map(r => r.resource_id);
        return [
          ...getTagsToInvalidateForImageMutation(image_ids),
          ...getTagsToInvalidateForVideoMutation(video_ids),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    deleteResources: build.mutation<
      paths['/api/v1/resources/delete']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/resources/delete']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildResourcesUrl('delete'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
        // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
        // will force those queries to re-fetch, and the requests will of course 404.
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
        ];
      },
    }),
    deleteUncategorizedResources: build.mutation<
      paths['/api/v1/resources/uncategorized']['delete']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({ url: buildResourcesUrl('uncategorized'), method: 'DELETE' }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
        // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
        // will force those queries to re-fetch, and the requests will of course 404.
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
        ];
      },
    }),
     /**
     * Star a list of images.
     */
     starResources: build.mutation<
     paths['/api/v1/resources/star']['post']['responses']['200']['content']['application/json'],
     paths['/api/v1/resources/star']['post']['requestBody']['content']['application/json']
   >({
     query: (body) => ({
       url: buildResourcesUrl('star'),
       method: 'POST',
       body,
     }),
     invalidatesTags: (result) => {
       if (!result) {
         return [];
       }
       const image_ids = result.starred_resources.filter(r => r.resource_type === "image").map(r => r.resource_id);
       const video_ids = result.starred_resources.filter(r => r.resource_type === "video").map(r => r.resource_id);
       return [
         ...getTagsToInvalidateForImageMutation(image_ids),
         ...getTagsToInvalidateForVideoMutation(video_ids),
         ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
         'ImageCollectionCounts',
         { type: 'ImageCollection', id: 'starred' },
         { type: 'ImageCollection', id: 'unstarred' },
       ];
     },
   }),
   /**
    * Unstar a list of images.
    */
   unstarResources: build.mutation<
     paths['/api/v1/resources/unstar']['post']['responses']['200']['content']['application/json'],
     paths['/api/v1/resources/unstar']['post']['requestBody']['content']['application/json']
   >({
     query: (body) => ({
       url: buildResourcesUrl('unstar'),
       method: 'POST',
       body,
     }),
     invalidatesTags: (result) => {
       if (!result) {
         return [];
       }
       const image_ids = result.unstarred_resources.filter(r => r.resource_type === "image").map(r => r.resource_id);
       const video_ids = result.unstarred_resources.filter(r => r.resource_type === "video").map(r => r.resource_id);
       return [
         ...getTagsToInvalidateForImageMutation(image_ids),
         ...getTagsToInvalidateForVideoMutation(video_ids),
         ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
         'ImageCollectionCounts',
         { type: 'ImageCollection', id: 'starred' },
         { type: 'ImageCollection', id: 'unstarred' },
       ];
     },
   }),
  }),
});

export const {
  useAddResourcesToBoardMutation,
  useRemoveResourcesFromBoardMutation,
  useDeleteResourcesMutation,
  useDeleteUncategorizedResourcesMutation,
  useStarResourcesMutation,
  useUnstarResourcesMutation,
} = resourcesApi;

