import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import { getListImagesUrl } from 'services/api/util';

import type { ApiTagDescription } from '..';

export const getTagsToInvalidateForBoardAffectingMutation = (affected_boards: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = ['ImageNameList'];

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
      type: 'VideoIdList',
    });

    tags.push({
      type: 'Board',
      id: board_id,
    });

    tags.push({
      type: 'BoardImagesTotal',
      id: board_id,
    });
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

export const getTagsToInvalidateForVideoMutation = (video_ids: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = [];

  for (const video_id of video_ids) {
    tags.push({
      type: 'Video',
      id: video_id,
    });
    // tags.push({
    //     type: 'VideoMetadata',
    //     id: video_id,
    // });
  }

  return tags;
};
