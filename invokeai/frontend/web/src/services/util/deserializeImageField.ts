import { Image } from 'app/invokeai';
import { ImageField, ImageType } from 'services/api';

export const buildImageUrls = (
  imageType: ImageType,
  imageName: string
): { url: string; thumbnail: string } => {
  const url = `api/v1/images/${imageType}/${imageName}`;

  const thumbnail = `api/v1/images/${imageType}/thumbnails/${
    imageName.split('.')[0]
  }.webp`;

  return {
    url,
    thumbnail,
  };
};

export const extractTimestampFromImageName = (imageName: string) => {
  const timestamp = imageName.split('_')?.pop()?.split('.')[0];

  if (timestamp === undefined) {
    return 0;
  }

  return Number(timestamp);
};

/**
 * Process ImageField objects. These come from `invocation_complete` events and do not contain all the data we need.
 * This is a WIP on the server side.
 */
export const deserializeImageField = (image: ImageField): Image => {
  const name = image.image_name;
  const type = image.image_type;

  const { url, thumbnail } = buildImageUrls(type, name);

  const timestamp = extractTimestampFromImageName(name);

  return {
    name,
    type,
    url,
    thumbnail,
    metadata: {
      timestamp,
      height: 512, // TODO: need the server to give this to us
      width: 512,
    },
  };
};
