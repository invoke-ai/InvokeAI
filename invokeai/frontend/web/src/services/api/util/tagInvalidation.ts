import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { getListImagesUrl } from 'services/api/util';

import type { ApiTagDescription } from '..';

export const getTagsToInvalidateForBoardAffectingMutation = (affected_boards: string[]): ApiTagDescription[] => {
  // Whenever an image or video mutation changes a board's contents we also have to refresh
  // the polymorphic gallery list (and its names companion) since that is what the gallery UI
  // actually subscribes to once Phase 4 lands.
  const tags: ApiTagDescription[] = [
    'ImageNameList',
    'VirtualBoards',
    'VideoNameList',
    'GalleryItemList',
    'GalleryItemNameList',
  ];

  for (const board_id of affected_boards) {
    tags.push({
      type: 'ImageList',
      id: getListImagesUrl({
        board_id,
        categories: IMAGE_CATEGORIES,
      }),
    });

    tags.push({
      type: 'ImageList',
      id: getListImagesUrl({
        board_id,
        categories: ASSETS_CATEGORIES,
      }),
    });

    tags.push({
      type: 'Board',
      id: board_id,
    });

    tags.push({
      type: 'BoardImagesTotal',
      id: board_id,
    });

    tags.push({
      type: 'BoardVideosTotal',
      id: board_id,
    });

    tags.push({
      type: 'BoardCanvasProjectsTotal',
      id: board_id,
    });
  }

  return tags;
};

export const getTagsToInvalidateForCanvasProjectMutation = (project_names: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = [];

  for (const project_name of project_names) {
    tags.push({ type: 'CanvasProject', id: project_name });
  }

  return tags;
};

export const getTagsToInvalidateForVideoMutation = (video_names: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = [];

  for (const video_name of video_names) {
    tags.push({ type: 'Video', id: video_name });
    tags.push({ type: 'VideoMetadata', id: video_name });
  }

  return tags;
};

export const getTagsToInvalidateForImageMutation = (image_names: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = [];

  for (const image_name of image_names) {
    tags.push({
      type: 'Image',
      id: image_name,
    });
    tags.push({
      type: 'ImageMetadata',
      id: image_name,
    });
    tags.push({
      type: 'ImageWorkflow',
      id: image_name,
    });
  }

  return tags;
};
