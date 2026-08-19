import type { FetchArgs, FetchBaseQueryError, QueryReturnValue } from '@reduxjs/toolkit/query';
import { skipToken } from '@reduxjs/toolkit/query';
import { getStore } from 'app/store/nanostores/store';
import { uniq } from 'es-toolkit';
import type { CroppableImageWithDims } from 'features/controlLayers/store/types';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { toast } from 'features/toast/toast';
import i18n from 'i18next';
import type { components, paths } from 'services/api/schema';
import type {
  GetImageNamesArgs,
  GetImageNamesResult,
  GraphAndWorkflowResponse,
  ImageDTO,
  ImageUploadEntryRequest,
  ImageUploadEntryResponse,
  ListImagesArgs,
  ListImagesResponse,
  UploadImageArg,
} from 'services/api/types';
import { getListImagesUrl } from 'services/api/util';
import {
  getTagsToInvalidateForBoardAffectingMutation,
  getTagsToInvalidateForImageMutation,
  getTagsToInvalidateForVideoMutation,
} from 'services/api/util/tagInvalidation';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';
import type { JsonObject } from 'type-fest';

import { api, buildV1Url, LIST_TAG } from '..';
import { buildBoardsUrl } from './boards';

/**
 * Builds an endpoint URL for the images router
 * @example
 * buildImagesUrl('some-path')
 * // '/api/v1/images/some-path'
 */
const buildImagesUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`images/${path}`, query);

/**
 * Builds an endpoint URL for the board_images router
 * @example
 * buildBoardImagesUrl('some-path')
 * // '/api/v1/board_images/some-path'
 */
const buildBoardImagesUrl = (path: string = '') => buildV1Url(`board_images/${path}`);

/**
 * The batch routes cap `image_names` server-side (`MAX_IMAGE_BATCH_SIZE` in
 * `invokeai/app/api/routers/images.py`) so one authenticated client cannot pin a worker with a
 * single request. Nothing caps a gallery *selection*: select-all reads the whole board's name
 * list, so one keystroke on a large board produces a selection an order of magnitude past the
 * cap. Oversized bodies are therefore split client-side into conforming requests.
 *
 * Must stay <= the backend constant.
 */
const IMAGE_BATCH_CHUNK_SIZE = 1000;

/**
 * Every batch route answers with an object whose values are all name lists, and every one of
 * them reports its per-name failures in `failed_images` — which is where a chunked run folds
 * the names it never reached.
 */
type ImageBatchResult = Record<string, string[]> & { failed_images: string[] };

type InvalidateTagsArg = Parameters<typeof api.util.invalidateTags>[0];

/** The `baseQuery` handed to a `queryFn`, matching what `fetchBaseQuery` produces. */
type ImagesBaseQuery = (
  args: string | FetchArgs
) => QueryReturnValue<unknown, FetchBaseQueryError> | PromiseLike<QueryReturnValue<unknown, FetchBaseQueryError>>;

export const chunkImageNames = (image_names: string[]): string[][] => {
  if (image_names.length <= IMAGE_BATCH_CHUNK_SIZE) {
    // Single-request path, byte-for-byte what it was before chunking existed. Note this also
    // covers the empty list: the routes answer it with a well-formed empty result, which
    // merging zero chunks could not reproduce.
    return [image_names];
  }
  const chunks: string[][] = [];
  for (let i = 0; i < image_names.length; i += IMAGE_BATCH_CHUNK_SIZE) {
    chunks.push(image_names.slice(i, i + IMAGE_BATCH_CHUNK_SIZE));
  }
  return chunks;
};

/**
 * Unions the per-chunk results key-by-key, so callers and `invalidatesTags` see one aggregate
 * result with the same shape a single request would have returned. Keys are unioned rather
 * than enumerated because the five batch routes name their outcome list differently
 * (`deleted_images`, `starred_images`, `added_images`, ...) while sharing `affected_boards`.
 */
export const mergeImageBatchResults = <TResult extends ImageBatchResult>(results: TResult[]): TResult => {
  // Accumulator is the looser Record: it is only complete once every chunk has contributed.
  const merged: Record<string, string[]> = {};
  for (const result of results) {
    for (const [key, names] of Object.entries(result)) {
      merged[key] = merged[key] ? uniq(merged[key].concat(names)) : names;
    }
  }
  return merged as TResult;
};

/**
 * Builds a `queryFn` that runs a batch mutation one conforming chunk at a time.
 *
 * Chunks go sequentially on purpose. Each is already up to `IMAGE_BATCH_CHUNK_SIZE` rows of DB
 * work, and the server-side bound exists to stop a single client from pinning a worker — firing
 * the chunks concurrently would hand straight back what the bound took away.
 *
 * The mid-run failure is the case that matters, and it resolves to a partial success rather than
 * an error. Earlier chunks have already been committed by the server; returning a bare error
 * would discard their payload, which is exactly the bug the partial-failure reporting on these
 * routes exists to fix. It is not only the RTK cache at stake — `handleDeletions` drives the
 * gallery selection and strips deleted images out of nodes, canvas layers and reference images
 * off `result.deleted_images`, and none of that runs on a rejection. So the merged result is
 * returned, the unreached names are folded into its `failed_images`, and only a run where
 * *nothing* landed surfaces as an error.
 *
 * They are folded into the payload rather than toasted here so that a single `onQueryStarted`
 * reports one total. Toasting from both places would fire twice on the same toast id, and the
 * toast system updates in place and appends "(2)" — so the second count would simply replace
 * the first rather than adding to it.
 */
export const buildChunkedImageBatchQueryFn =
  <TResult extends ImageBatchResult, TArg extends { image_names: string[] }>(
    request: (body: TArg) => { url: string; method: string },
    getTags: (result: TResult) => InvalidateTagsArg
  ) =>
  async (
    arg: TArg,
    { dispatch }: { dispatch: (action: ReturnType<typeof api.util.invalidateTags>) => unknown },
    _extraOptions: unknown,
    baseQuery: ImagesBaseQuery
  ) => {
    const results: TResult[] = [];
    const chunks = chunkImageNames(arg.image_names);
    for (const [index, image_names] of chunks.entries()) {
      const response = await baseQuery({ ...request(arg), body: { ...arg, image_names } });
      if (response.error) {
        if (results.length === 0) {
          // Nothing was applied, so this is an ordinary failed request — report it as one.
          return { error: response.error };
        }
        // Everything from this chunk on is unreached, not merely un-reported.
        const unreached = chunks.slice(index).flat();
        const merged = mergeImageBatchResults(results);
        dispatch(api.util.invalidateTags(getTags(merged)));
        return { data: { ...merged, failed_images: uniq(merged.failed_images.concat(unreached)) } as TResult };
      }
      results.push(response.data as TResult);
    }
    return { data: mergeImageBatchResults(results) };
  };

/**
 * Tag sets for the chunked batch mutations. Extracted from the endpoints so the chunked
 * `queryFn` can invalidate for the chunks that landed before a mid-run failure using the exact
 * tag set the endpoint publishes on success, rather than a hand-rolled approximation.
 */
const getDeleteImagesTags = (result: components['schemas']['DeleteImagesResult']): InvalidateTagsArg => [
  // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
  // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
  // will force those queries to re-fetch, and the requests will of course 404.
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
  'ImageCollectionCounts',
  { type: 'ImageCollection', id: LIST_TAG },
];

const getStarImagesTags = (result: components['schemas']['StarredImagesResult']): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.starred_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
  'ImageCollectionCounts',
  { type: 'ImageCollection', id: 'starred' },
  { type: 'ImageCollection', id: 'unstarred' },
];

const getUnstarImagesTags = (result: components['schemas']['UnstarredImagesResult']): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.unstarred_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
  'ImageCollectionCounts',
  { type: 'ImageCollection', id: 'starred' },
  { type: 'ImageCollection', id: 'unstarred' },
];

const getAddImagesToBoardTags = (result: components['schemas']['AddImagesToBoardResult']): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.added_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
];

const getRemoveImagesFromBoardTags = (
  result: components['schemas']['RemoveImagesFromBoardResult']
): InvalidateTagsArg => [
  ...getTagsToInvalidateForImageMutation(result.removed_images),
  ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
];

/**
 * Surfaces the partial-failure warning. Used for both kinds of partial failure: the names the
 * server reported it could not apply, and the names a chunked run never reached.
 */
const toastFailedImages = (count: number) => {
  if (count > 0) {
    toast({
      id: 'IMAGES_FAILED_TO_UPDATE',
      title: i18n.t('toast.imagesFailedToUpdate', { count }),
      status: 'warning',
    });
  }
};

/**
 * The download counterpart. Distinct id and wording: "could not be updated" is wrong for a
 * download, and sharing the id would let one warning overwrite the other, since the toast
 * system updates in place.
 */
const toastFailedDownloads = (count: number) => {
  if (count > 0) {
    toast({
      id: 'IMAGES_FAILED_TO_DOWNLOAD',
      title: i18n.t('toast.imagesFailedToDownload', { count }),
      status: 'warning',
    });
  }
};

/**
 * Runs `/images/download` one conforming chunk at a time.
 *
 * Chunked like the mutating batch routes, but it cannot merge: each request produces its own
 * bulk-download item, so an oversized selection becomes several zips rather than one. That is
 * the cost of keeping the selection downloadable at all — the alternative is a 422 the moment
 * the user hits select-all on a board past the cap.
 *
 * Only the first item name is returned, and only to give `matchFulfilled` a name for the single
 * "preparing" toast. The zips themselves arrive independently: each background task emits its
 * own `bulk_download_complete`, and the socket handler fetches and saves per event, keyed on the
 * item name in the event rather than on this payload.
 *
 * A mid-run failure therefore cannot be reported as a plain rejection. The route answers 202 the
 * moment it has scheduled the background task, so every chunk before the failing one is already
 * producing a zip that will land in the user's downloads. Rejecting drives `matchRejected` and
 * toasts "problem preparing download" while those zips arrive anyway — the opposite of what
 * happened. So, exactly as in `buildChunkedImageBatchQueryFn`, only a run where *nothing* was
 * scheduled surfaces as an error; a partial run resolves and reports the names that never made
 * it into any zip.
 */
export const bulkDownloadQueryFn = async (
  { image_names, board_id }: components['schemas']['Body_download_images_from_list'],
  _api: unknown,
  _extraOptions: unknown,
  baseQuery: ImagesBaseQuery
) => {
  // A board download expands server-side from board_id alone, so there is nothing to split.
  // Note the server picks board_id over image_names when both are set (`BulkDownloadService`),
  // so a body carrying both is not a selection this can meaningfully chunk — no caller sends
  // one, and splitting it would ask for the same full-board zip once per chunk.
  const chunks = image_names?.length ? chunkImageNames(image_names) : [image_names ?? []];
  // Tracked separately from the payload rather than inferred from it. `fetchBaseQuery` resolves
  // an empty response entity as `data: null`, so a 202 whose body did not survive the trip back
  // leaves nothing to return even though the background task WAS scheduled and its zip is
  // already being built. Any nullish test over the payload then sends us down the
  // nothing-happened path and toasts a failure over an arriving download.
  let scheduled = false;
  let first: components['schemas']['ImagesDownloaded'] | undefined;
  for (const [index, chunk] of chunks.entries()) {
    const response = await baseQuery({
      url: buildImagesUrl('download'),
      method: 'POST',
      body: { image_names: chunk, board_id },
    });
    if (response.error) {
      if (!scheduled) {
        // Nothing was scheduled, so this is an ordinary failed request — report it as one and
        // let the `matchRejected` listener raise the failure toast.
        return { error: response.error };
      }
      // Everything from this chunk on is unreached: those names are in no zip. The failing
      // chunk counts too — it was reached, but nothing was scheduled for it. Toasted here
      // rather than folded into the payload because `ImagesDownloaded` has no per-name failure
      // list, and unlike the mutating routes this endpoint has no `onQueryStarted` that would
      // make this a second reporting site for the same toast id.
      toastFailedDownloads(chunks.slice(index).flat().length);
      return { data: first };
    }
    scheduled = true;
    first ??= response.data as components['schemas']['ImagesDownloaded'];
  }
  return { data: first as components['schemas']['ImagesDownloaded'] };
};

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    listImages: build.query<ListImagesResponse, ListImagesArgs>({
      query: (queryArgs) => ({
        // Use the helper to create the URL.
        url: getListImagesUrl(queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => {
        return [
          // Make the tags the same as the cache key
          { type: 'ImageList', id: stableHash(queryArgs) },
          { type: 'Board', id: queryArgs.board_id ?? 'none' },
          'FetchOnReconnect',
        ];
      },
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        // Populate the getImageDTO cache with these images. This makes image selection smoother, because it doesn't
        // need to re-fetch image data when the user selects an image. The getImageDTO cache keeps data for the default
        // of 60s, so this data won't stick around too long.
        const res = await queryFulfilled;
        const imageDTOs = res.data.items;
        const updates: Param0<typeof imagesApi.util.upsertQueryEntries> = [];
        for (const imageDTO of imageDTOs) {
          updates.push({
            endpointName: 'getImageDTO',
            arg: imageDTO.image_name,
            value: imageDTO,
          });
        }
        dispatch(imagesApi.util.upsertQueryEntries(updates));
      },
    }),
    getIntermediatesCount: build.query<number, void>({
      query: () => ({ url: buildImagesUrl('intermediates') }),
      providesTags: ['IntermediatesCount', 'FetchOnReconnect'],
    }),
    clearIntermediates: build.mutation<number, void>({
      query: () => ({ url: buildImagesUrl('intermediates'), method: 'DELETE' }),
      invalidatesTags: [
        'IntermediatesCount',
        'InvocationCacheStatus',
        'ImageCollectionCounts',
        { type: 'ImageCollection', id: LIST_TAG },
      ],
    }),
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}`) }),
      providesTags: (result, error, image_name) => [{ type: 'Image', id: image_name }],
    }),
    getImageMetadata: build.query<JsonObject | undefined, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}/metadata`) }),
      providesTags: (result, error, image_name) => [{ type: 'ImageMetadata', id: image_name }],
    }),
    getImageWorkflow: build.query<GraphAndWorkflowResponse, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}/workflow`) }),
      providesTags: (result, error, image_name) => [{ type: 'ImageWorkflow', id: image_name }],
    }),
    deleteImage: build.mutation<
      paths['/api/v1/images/i/{image_name}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/i/{image_name}']['delete']['parameters']['path']
    >({
      query: ({ image_name }) => ({
        url: buildImagesUrl(`i/${image_name}`),
        method: 'DELETE',
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
    deleteImages: build.mutation<
      paths['/api/v1/images/delete']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/delete']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildImagesUrl('delete'), method: 'POST' }),
        getDeleteImagesTags
      ),
      async onQueryStarted(_, { queryFulfilled }) {
        try {
          const { data: result } = await queryFulfilled;
          // `handleDeletions` swallows every outcome, so without this a delete that only
          // partly landed — server-side failures or chunks never reached — said nothing at all.
          toastFailedImages(result.failed_images.length);
        } catch {
          // A rejection means nothing landed at all -- a partial run resolves with data,
          // not an error. Nothing toasts these rejections today; that gap is unchanged
          // by this handler, which exists only to surface per-name failures.
        }
      },
      invalidatesTags: (result) => (result ? getDeleteImagesTags(result) : []),
    }),
    deleteUncategorizedImages: build.mutation<
      paths['/api/v1/images/uncategorized']['delete']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({ url: buildImagesUrl('uncategorized'), method: 'DELETE' }),
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
     * Change an image's `is_intermediate` property.
     */
    changeImageIsIntermediate: build.mutation<
      paths['/api/v1/images/i/{image_name}']['patch']['responses']['200']['content']['application/json'],
      { image_name: string; is_intermediate: boolean }
    >({
      query: ({ image_name, is_intermediate }) => ({
        url: buildImagesUrl(`i/${image_name}`),
        method: 'PATCH',
        body: { is_intermediate },
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation([result.image_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([result.board_id ?? 'none']),
        ];
      },
    }),
    /**
     * Star a list of images.
     */
    starImages: build.mutation<
      paths['/api/v1/images/star']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/star']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildImagesUrl('star'), method: 'POST' }),
        getStarImagesTags
      ),
      async onQueryStarted(_, { queryFulfilled }) {
        try {
          const { data: result } = await queryFulfilled;
          toastFailedImages(result.failed_images.length);
        } catch {
          // A rejection means nothing landed at all -- a partial run resolves with data,
          // not an error. Nothing toasts these rejections today; that gap is unchanged
          // by this handler, which exists only to surface per-name failures.
        }
      },
      invalidatesTags: (result) => (result ? getStarImagesTags(result) : []),
    }),
    /**
     * Unstar a list of images.
     */
    unstarImages: build.mutation<
      paths['/api/v1/images/unstar']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/unstar']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildImagesUrl('unstar'), method: 'POST' }),
        getUnstarImagesTags
      ),
      async onQueryStarted(_, { queryFulfilled }) {
        try {
          const { data: result } = await queryFulfilled;
          toastFailedImages(result.failed_images.length);
        } catch {
          // A rejection means nothing landed at all -- a partial run resolves with data,
          // not an error. Nothing toasts these rejections today; that gap is unchanged
          // by this handler, which exists only to surface per-name failures.
        }
      },
      invalidatesTags: (result) => (result ? getUnstarImagesTags(result) : []),
    }),
    uploadImage: build.mutation<
      paths['/api/v1/images/upload']['post']['responses']['201']['content']['application/json'],
      UploadImageArg
    >({
      query: ({ file, image_category, is_intermediate, session_id, board_id, crop_visible, metadata, resize_to }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }
        if (resize_to) {
          formData.append('resize_to', JSON.stringify(resize_to));
        }
        return {
          url: buildImagesUrl('upload'),
          method: 'POST',
          body: formData,
          params: {
            image_category,
            is_intermediate,
            session_id,
            board_id: board_id === 'none' ? undefined : board_id,
            crop_visible,
          },
        };
      },
      invalidatesTags: (result) => {
        if (!result || result.is_intermediate) {
          // Don't add it to anything
          return [];
        }
        const boardId = result.board_id ?? 'none';

        return [
          ...getTagsToInvalidateForImageMutation([result.image_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([boardId]),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
          'ImageNameList',
        ];
      },
    }),
    createImageUploadEntry: build.mutation<ImageUploadEntryResponse, ImageUploadEntryRequest>({
      query: ({ width, height, board_id }) => ({
        url: buildImagesUrl(),
        method: 'POST',
        body: { width, height, board_id },
      }),
    }),
    deleteBoard: build.mutation<
      paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/boards/{board_id}']['delete']['parameters']['path']
    >({
      query: ({ board_id }) => ({ url: buildBoardsUrl(board_id), method: 'DELETE' }),
      invalidatesTags: (result) => [
        { type: 'Board', id: LIST_TAG },
        // Both images and videos on the board cascade to the 'No Board' bucket on the
        // backend side; invalidate the 'none' caches for both kinds so the polymorphic
        // gallery surfaces them. The Gallery* tags refresh the unified gallery list view.
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: IMAGE_CATEGORIES,
          }),
        },
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: ASSETS_CATEGORIES,
          }),
        },
        { type: 'VideoList', id: LIST_TAG },
        'VideoNameList',
        'GalleryItemList',
        'GalleryItemNameList',
        // The orphaned media keep cached DTOs whose board_id still points at the deleted
        // board; refetch them so drag/drop and context menus see the new 'none' board.
        ...getTagsToInvalidateForImageMutation(result?.deleted_board_images ?? []),
        ...getTagsToInvalidateForVideoMutation(result?.deleted_board_videos ?? []),
      ],
    }),

    deleteBoardAndImages: build.mutation<
      paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/boards/{board_id}']['delete']['parameters']['path']
    >({
      query: ({ board_id }) => ({
        url: buildBoardsUrl(board_id),
        method: 'DELETE',
        params: { include_images: true },
      }),
      // The backend now also cascade-deletes videos on the board, so the unified gallery
      // and the video list both need invalidation in addition to the board tag.
      invalidatesTags: (result) => [
        { type: 'Board', id: LIST_TAG },
        'VirtualBoards',
        { type: 'VideoList', id: LIST_TAG },
        'VideoNameList',
        'GalleryItemList',
        'GalleryItemNameList',
        ...getTagsToInvalidateForBoardAffectingMutation(['none']),
        // Deleted media must drop out of their per-item caches (so DTO queries become
        // 404s instead of serving stale entries, e.g. to node inputs still referencing
        // them). Only server-confirmed deletions are listed in the result.
        ...getTagsToInvalidateForImageMutation(result?.deleted_images ?? []),
        ...getTagsToInvalidateForVideoMutation(result?.deleted_videos ?? []),
        ...getTagsToInvalidateForImageMutation(result?.failed_images ?? []),
        ...getTagsToInvalidateForVideoMutation(result?.failed_videos ?? []),
      ],
    }),
    addImageToBoard: build.mutation<
      paths['/api/v1/board_images/']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => {
        return {
          url: buildBoardImagesUrl(),
          method: 'POST',
          body,
        };
      },
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.added_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    removeImageFromBoard: build.mutation<
      paths['/api/v1/board_images/']['delete']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/']['delete']['requestBody']['content']['application/json']
    >({
      query: (body) => {
        return {
          url: buildBoardImagesUrl(),
          method: 'DELETE',
          body,
        };
      },
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.removed_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    addImagesToBoard: build.mutation<
      paths['/api/v1/board_images/batch']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/batch']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildBoardImagesUrl('batch'), method: 'POST' }),
        getAddImagesToBoardTags
      ),
      async onQueryStarted(_, { queryFulfilled }) {
        try {
          const { data: result } = await queryFulfilled;
          toastFailedImages(result.failed_images.length);
        } catch {
          // A rejection means nothing landed at all -- a partial run resolves with data,
          // not an error. Nothing toasts these rejections today; that gap is unchanged
          // by this handler, which exists only to surface per-name failures.
        }
      },
      invalidatesTags: (result) => (result ? getAddImagesToBoardTags(result) : []),
    }),
    removeImagesFromBoard: build.mutation<
      paths['/api/v1/board_images/batch/delete']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/batch/delete']['post']['requestBody']['content']['application/json']
    >({
      queryFn: buildChunkedImageBatchQueryFn(
        () => ({ url: buildBoardImagesUrl('batch/delete'), method: 'POST' }),
        getRemoveImagesFromBoardTags
      ),
      async onQueryStarted(_, { queryFulfilled }) {
        try {
          const { data: result } = await queryFulfilled;
          toastFailedImages(result.failed_images.length);
        } catch {
          // A rejection means nothing landed at all -- a partial run resolves with data,
          // not an error. Nothing toasts these rejections today; that gap is unchanged
          // by this handler, which exists only to surface per-name failures.
        }
      },
      invalidatesTags: (result) => (result ? getRemoveImagesFromBoardTags(result) : []),
    }),
    bulkDownloadImages: build.mutation<
      components['schemas']['ImagesDownloaded'],
      components['schemas']['Body_download_images_from_list']
    >({
      queryFn: bulkDownloadQueryFn,
    }),
    /**
     * Get ordered list of image names for selection operations
     */
    getImageNames: build.query<GetImageNamesResult, GetImageNamesArgs>({
      query: (queryArgs) => ({
        url: buildImagesUrl('names', queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        'ImageNameList',
        'FetchOnReconnect',
        { type: 'ImageNameList', id: stableHash(queryArgs) },
      ],
    }),
    /**
     * Get image DTOs for the specified image names. Maintains order of input names.
     */
    getImageDTOsByNames: build.mutation<
      paths['/api/v1/images/images_by_names']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/images_by_names']['post']['requestBody']['content']['application/json']
    >({
      /**
       * Chunked too, despite being a read: `useRangeBasedImageFetching` unions every virtuoso
       * range seen inside its throttle window, and a dense grid with a 4096px overscan can carry
       * that past the cap. The failure was silent — the hook swallows the rejection and nothing
       * listens for it, so the affected thumbnails simply never loaded. Results concatenate
       * because the route returns a plain ordered list, and the caller only upserts by name.
       */
      queryFn: async ({ image_names }, _api, _extraOptions, baseQuery) => {
        const imageDTOs: ImageDTO[] = [];
        for (const chunk of chunkImageNames(image_names)) {
          const response = await baseQuery({
            url: buildImagesUrl('images_by_names'),
            method: 'POST',
            body: { image_names: chunk },
          });
          if (response.error) {
            return { error: response.error };
          }
          imageDTOs.push(...(response.data as ImageDTO[]));
        }
        return { data: imageDTOs };
      },
      // Don't provide cache tags - we'll manually upsert into individual getImageDTO caches
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        try {
          const { data: imageDTOs } = await queryFulfilled;

          // Upsert each DTO into the individual image cache
          const updates: Param0<typeof imagesApi.util.upsertQueryEntries> = [];
          for (const imageDTO of imageDTOs) {
            updates.push({
              endpointName: 'getImageDTO',
              arg: imageDTO.image_name,
              value: imageDTO,
            });
          }
          dispatch(imagesApi.util.upsertQueryEntries(updates));
        } catch {
          // Handle error if needed
        }
      },
    }),
  }),
});

export const {
  useGetIntermediatesCountQuery,
  useListImagesQuery,
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
  useGetImageWorkflowQuery,
  useLazyGetImageWorkflowQuery,
  useUploadImageMutation,
  useClearIntermediatesMutation,
  useAddImagesToBoardMutation,
  useRemoveImagesFromBoardMutation,
  useDeleteBoardAndImagesMutation,
  useDeleteUncategorizedImagesMutation,
  useDeleteBoardMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
  useBulkDownloadImagesMutation,
  useGetImageDTOsByNamesMutation,
} = imagesApi;

/**
 * Imperative RTKQ helper to fetch an ImageDTO.
 * @param image_name The name of the image to fetch
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @returns The ImageDTO if found, otherwise null
 */
export const getImageDTOSafe = async (
  image_name: string,
  options?: Parameters<typeof imagesApi.endpoints.getImageDTO.initiate>[1]
): Promise<ImageDTO | null> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(imagesApi.endpoints.getImageDTO.initiate(image_name, _options));
  try {
    return await req.unwrap();
  } catch {
    return null;
  }
};

/**
 * Imperative RTKQ helper to fetch an ImageDTO.
 * @param image_name The name of the image to fetch
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @raises Error if the image is not found or there is an error fetching the image
 */
export const getImageDTO = (
  image_name: string,
  options?: Parameters<typeof imagesApi.endpoints.getImageDTO.initiate>[1]
): Promise<ImageDTO> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(imagesApi.endpoints.getImageDTO.initiate(image_name, _options));
  return req.unwrap();
};

export const uploadImage = (arg: UploadImageArg): Promise<ImageDTO> => {
  const { dispatch } = getStore();
  const req = dispatch(imagesApi.endpoints.uploadImage.initiate(arg, { track: false }));
  return req.unwrap();
};

export const copyImage = async (imageName: string, uploadImageArg: Omit<UploadImageArg, 'file'>): Promise<ImageDTO> => {
  const originalImageDTO = await getImageDTO(imageName);
  const file = await imageDTOToFile(originalImageDTO);
  const imageDTO = await uploadImage({ file, ...uploadImageArg });
  return imageDTO;
};

export const uploadImages = async (args: UploadImageArg[]): Promise<ImageDTO[]> => {
  const { dispatch } = getStore();
  const results = await Promise.allSettled(
    args.map((arg) => {
      const req = dispatch(imagesApi.endpoints.uploadImage.initiate(arg, { track: false }));
      return req.unwrap();
    })
  );
  return results.filter((r): r is PromiseFulfilledResult<ImageDTO> => r.status === 'fulfilled').map((r) => r.value);
};

/**
 * Convert an ImageDTO to a File by downloading the image from the server.
 * @param imageDTO The image to download and convert to a File
 */
export const imageDTOToFile = async (imageDTO: ImageDTO): Promise<File> => {
  const init: RequestInit = {};
  const res = await fetch(imageDTO.image_url, init);
  const blob = await res.blob();
  // Create a new file with the same name, which we will upload
  const file = new File([blob], `copy_of_${imageDTO.image_name}`, { type: 'image/png' });
  return file;
};

export const useImageDTO = (imageName: string | null | undefined) => {
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);
  return imageDTO ?? null;
};

export const useImageDTOFromCroppableImage = (croppableImage: CroppableImageWithDims | null) => {
  const { currentData: imageDTO } = useGetImageDTOQuery(
    croppableImage?.crop?.image.image_name ?? croppableImage?.original.image.image_name ?? skipToken
  );
  return imageDTO ?? null;
};
