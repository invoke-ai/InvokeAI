import type { ImageDTO, VideoDTO } from 'services/api/types';

// Type guards
export function isImageResource(resource: ImageDTO | VideoDTO): resource is ImageDTO {
  return 'image_name' in resource;
}

export function isVideoResource(resource: ImageDTO | VideoDTO): resource is VideoDTO {
  return 'video_id' in resource;
}

