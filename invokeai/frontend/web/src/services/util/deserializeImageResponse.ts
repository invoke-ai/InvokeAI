import { Image } from 'app/invokeai';
import { ImageResponse } from 'services/api';

/**
 * Process ImageReponse objects, which we get from the `list_images` endpoint.
 */
export const deserializeImageResponse = (
  imageResponse: ImageResponse
): Image => {
  const { image_name, image_type, image_url, metadata, thumbnail_url } =
    imageResponse;

  // TODO: parse metadata - just leaving it as-is for now

  return {
    name: image_name,
    type: image_type,
    url: image_url,
    thumbnail: thumbnail_url,
    metadata,
  };
};
