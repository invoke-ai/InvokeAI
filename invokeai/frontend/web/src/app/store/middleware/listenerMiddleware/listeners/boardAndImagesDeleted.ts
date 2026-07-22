import { isAnyOf } from '@reduxjs/toolkit';
import type { AppStartListening } from 'app/store/store';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { getImageUsage } from 'features/deleteImageModal/store/state';
import { clearNodesVideoFields } from 'features/deleteVideoModal/store/state';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { selectUpscaleSlice } from 'features/parameters/store/upscaleSlice';
import { api } from 'services/api';
import { imagesApi } from 'services/api/endpoints/images';
import { videosApi } from 'services/api/endpoints/videos';
import {
  getTagsToInvalidateForImageMutation,
  getTagsToInvalidateForVideoMutation,
} from 'services/api/util/tagInvalidation';

const getDeletedNamesFromDeleteBoardAction = (payload: unknown, key: 'deleted_images' | 'deleted_videos'): string[] => {
  if (!payload || typeof payload !== 'object') {
    return [];
  }
  const response = payload as Record<string, unknown> & { data?: unknown };
  let deletedNames = response[key];
  if (deletedNames === undefined && response.data && typeof response.data === 'object') {
    const detail = (response.data as { detail?: unknown }).detail;
    if (detail && typeof detail === 'object') {
      deletedNames = (detail as Record<string, unknown>)[key];
    }
  }
  return Array.isArray(deletedNames) && deletedNames.every((name) => typeof name === 'string') ? deletedNames : [];
};

export const getDeletedImagesFromDeleteBoardAction = (payload: unknown): string[] =>
  getDeletedNamesFromDeleteBoardAction(payload, 'deleted_images');

export const getDeletedVideosFromDeleteBoardAction = (payload: unknown): string[] =>
  getDeletedNamesFromDeleteBoardAction(payload, 'deleted_videos');

export const addDeleteBoardAndImagesFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: isAnyOf(
      imagesApi.endpoints.deleteBoardAndImages.matchFulfilled,
      imagesApi.endpoints.deleteBoardAndImages.matchRejected
    ),
    effect: (action, { dispatch, getState }) => {
      const deletedImages = getDeletedImagesFromDeleteBoardAction(action.payload);
      const deletedVideos = getDeletedVideosFromDeleteBoardAction(action.payload);

      // On a *partial* board deletion the mutation rejects, so its invalidatesTags run
      // with no result and only the static list tags fire — the per-item caches
      // (Image/ImageMetadata/ImageWorkflow, Video/...) of the confirmed-deleted media
      // would otherwise stay readable. We already parsed the confirmed names out of the
      // 500's detail above, so invalidate them here (harmlessly redundant on success).
      const itemTags = [
        ...getTagsToInvalidateForImageMutation(deletedImages),
        ...getTagsToInvalidateForVideoMutation(deletedVideos),
      ];
      if (itemTags.length > 0) {
        dispatch(api.util.invalidateTags(itemTags));
      }

      // Remove all deleted images from the UI

      let wasNodeEditorReset = false;

      const state = getState();
      const nodes = selectNodesSlice(state);
      const canvas = selectCanvasSlice(state);
      const upscale = selectUpscaleSlice(state);
      const refImages = selectRefImagesSlice(state);

      deletedImages.forEach((image_name) => {
        const imageUsage = getImageUsage(nodes, canvas, upscale, refImages, image_name);

        if (imageUsage.isNodesImage && !wasNodeEditorReset) {
          dispatch(nodeEditorReset());
          wasNodeEditorReset = true;
        }
      });

      // Clear workflow-node VideoField inputs that referenced a cascade-deleted video —
      // same cleanup the direct video-delete flow performs. Skipped if the image sweep
      // already reset the node editor.
      if (!wasNodeEditorReset) {
        deletedVideos.forEach((video_name) => {
          clearNodesVideoFields(getState(), dispatch, video_name);
        });
      }
    },
  });

  // "Delete All Uncategorized Images/Videos" deletes videos through this dedicated
  // endpoint — its confirmed deletions need the same VideoField cleanup.
  startAppListening({
    matcher: videosApi.endpoints.deleteUncategorizedVideos.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      action.payload.deleted_videos.forEach((video_name) => {
        clearNodesVideoFields(getState(), dispatch, video_name);
      });
    },
  });
};
