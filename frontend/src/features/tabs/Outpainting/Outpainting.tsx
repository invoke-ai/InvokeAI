import React from 'react';
import { RootState, useAppSelector } from '../../../app/store';
import ImageGallery from '../../gallery/ImageGallery';
import OutpaintingImageDisplay from '../../gallery/OutpaintingImageDisplay';
import OutpaintingPanel from './OutpaintingPanel';

export default function Outpainting() {
  const shouldShowGallery = useAppSelector(
    (state: RootState) => state.options.shouldShowGallery
  );

  return (
    <div className="outpainting-workarea">
      <OutpaintingPanel />
      <div className="outpainting-display" style={
        shouldShowGallery
          ? { gridTemplateColumns: 'auto max-content' }
          : { gridTemplateColumns: 'auto' }
      }>
        <OutpaintingImageDisplay />
        <ImageGallery />
      </div>
    </div>
  );
}
