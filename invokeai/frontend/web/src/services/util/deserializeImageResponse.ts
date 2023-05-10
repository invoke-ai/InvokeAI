import { Image } from 'app/types/invokeai';
import { parseInvokeAIMetadata } from 'common/util/parseMetadata';
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
  const { invokeai, ...rest } = metadata;

  const parsedMetadata = parseInvokeAIMetadata(invokeai);

  return {
    name: image_name,
    type: image_type,
    url: image_url,
    thumbnail: thumbnail_url,
    metadata: {
      ...rest,
      ...(invokeai ? { invokeai: parsedMetadata } : {}),
    },
  };
};
