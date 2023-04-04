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

export const processImageField = (image: ImageField): Image => {
  const name = image.image_name;
  const type = image.image_type;

  const { url, thumbnail } = buildImageUrls(type, name);

  const timestamp = extractTimestampFromImageName(name);

  return {
    name,
    url,
    thumbnail,
    timestamp,
    height: 512,
    width: 512,
  };
};
