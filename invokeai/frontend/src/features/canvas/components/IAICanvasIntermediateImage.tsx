import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import { ImageConfig } from 'konva/lib/shapes/Image';
import { isEqual } from 'lodash';

import { useEffect, useState } from 'react';
import { Image as KonvaImage } from 'react-konva';

const selector = createSelector(
  [(state: RootState) => state.gallery],
  (gallery: GalleryState) => {
    return gallery.intermediateImage ? gallery.intermediateImage : null;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type Props = Omit<ImageConfig, 'image'>;

const IAICanvasIntermediateImage = (props: Props) => {
  const { ...rest } = props;
  const intermediateImage = useAppSelector(selector);

  const [loadedImageElement, setLoadedImageElement] =
    useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!intermediateImage) return;
    const tempImage = new Image();

    tempImage.onload = () => {
      setLoadedImageElement(tempImage);
    };
    tempImage.src = intermediateImage.url;
  }, [intermediateImage]);

  if (!intermediateImage?.boundingBox) return null;

  const {
    boundingBox: { x, y, width, height },
  } = intermediateImage;

  return loadedImageElement ? (
    <KonvaImage
      x={x}
      y={y}
      width={width}
      height={height}
      image={loadedImageElement}
      listening={false}
      {...rest}
    />
  ) : null;
};

export default IAICanvasIntermediateImage;
