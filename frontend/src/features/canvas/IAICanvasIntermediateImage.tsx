import { createSelector } from '@reduxjs/toolkit';
import { RootState, useAppSelector } from 'app/store';
import { GalleryState } from 'features/gallery/gallerySlice';
import _ from 'lodash';
import { Image } from 'react-konva';
import useImage from 'use-image';
import { currentCanvasSelector } from './canvasSlice';

const selector = createSelector(
  [currentCanvasSelector, (state: RootState) => state.gallery],
  (currentCanvas, gallery: GalleryState) => {
    const { x, y } = currentCanvas.boundingBoxCoordinates;
    const { width, height } = currentCanvas.boundingBoxDimensions;
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

const IAICanvasIntermediateImage = () => {
  const { x, y, width, height, url } = useAppSelector(selector);
  const [image] = useImage(url);
  return (
    <Image
      x={x}
      y={y}
      width={width}
      height={height}
      image={image}
      visible={Boolean(url)}
      listening={false}
    />
  );
};

export default IAICanvasIntermediateImage;
