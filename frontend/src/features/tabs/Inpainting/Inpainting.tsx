import React from 'react';
import ImageGallery from '../../gallery/ImageGallery';

import { RootState, useAppSelector } from '../../../app/store';
import ImageToImagePanel from '../ImageToImage/ImageToImagePanel';
import PaintingImageDisplay from '../../gallery/PaintingImageDisplay';

export default function Inpainting() {
  const shouldShowGallery = useAppSelector(
    (state: RootState) => state.options.shouldShowGallery
  );

  return (
    <div className="outpainting-workarea">
      <ImageToImagePanel />
      <div
        className="outpainting-display"
        style={
          shouldShowGallery
            ? { gridTemplateColumns: 'auto max-content' }
            : { gridTemplateColumns: 'auto' }
        }
      >
        <PaintingImageDisplay />
        <ImageGallery />
      </div>
    </div>
  );
}
