import { createSelector } from '@reduxjs/toolkit';
import { RootState, useAppSelector } from 'app/store';
import { GalleryState } from 'features/gallery/gallerySlice';
import { ImageConfig } from 'konva/lib/shapes/Image';
import _ from 'lodash';
import { useEffect, useState } from 'react';
import { Image as KonvaImage } from 'react-konva';
import { currentCanvasSelector } from './canvasSlice';

const selector = createSelector(
  [currentCanvasSelector, (state: RootState) => state.gallery],
  (currentCanvas, gallery: GalleryState) => {
    const {
      boundingBoxCoordinates: { x, y },
      boundingBoxDimensions: { width, height },
    } = currentCanvas;
    return {
      x,
      y,
      width,
      height,
      url: gallery.intermediateImage ? gallery.intermediateImage.url : '',
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

type Props = Omit<ImageConfig, 'image'>;

const IAICanvasIntermediateImage = (props: Props) => {
  const { ...rest } = props;
  const { x, y, width, height, url } = useAppSelector(selector);

  const [loadedImageElement, setLoadedImageElement] =
    useState<HTMLImageElement | null>(null);

  useEffect(() => {
    const tempImage = new Image();

    tempImage.onload = () => {
      setLoadedImageElement(tempImage);
    };
    tempImage.src = url;
  }, [url]);

  return url && loadedImageElement ? (
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
