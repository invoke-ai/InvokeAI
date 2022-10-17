import React from 'react';
import ImageGallery from '../../gallery/ImageGallery';

import { RootState, useAppSelector } from '../../../app/store';
import ImageToImagePanel from '../ImageToImage/ImageToImagePanel';
import InpaintingEditor from './InpaintingEditor';

export default function Inpainting() {
  const shouldShowGallery = useAppSelector(
    (state: RootState) => state.options.shouldShowGallery
  );

  return (
    <div className="image-to-image-workarea">
      <ImageToImagePanel />
      <div
        className="image-to-image-display-area"
        style={
          shouldShowGallery
            ? { gridTemplateColumns: 'auto max-content' }
            : { gridTemplateColumns: 'auto' }
        }
      >
        <InpaintingEditor />
        <ImageGallery />
      </div>
    </div>
  );
}
