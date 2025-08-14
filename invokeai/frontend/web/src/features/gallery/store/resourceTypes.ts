import type { ImageDTO, VideoDTO } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

// Type guards
export function isImageResource(resource: ImageDTO | VideoDTO): resource is ImageDTO {
  return resource.type === 'image';
}

export function isVideoResource(resource: ImageDTO | VideoDTO): resource is VideoDTO {
  return resource.type === 'video';
}

export const getResourceId = (resource: ImageDTO | VideoDTO): string => {
  switch (resource.type) {
    case 'image':
      return resource.image_name;
    case 'video':
      return resource.video_id;
    default:
      assert<Equals<never, typeof resource>>(false);
  }
};
