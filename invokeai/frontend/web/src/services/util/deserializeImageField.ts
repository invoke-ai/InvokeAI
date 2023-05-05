import { Image } from 'app/types/invokeai';
import { ImageField, ImageType } from 'services/api';
import { AnyInvocation } from 'services/events/types';

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
