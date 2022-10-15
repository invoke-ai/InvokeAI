import React from 'react';
import OutpaintingImageDisplay from '../../gallery/OutpaintingImageDisplay';
import OutpaintingPanel from './OutpaintingPanel';

export default function Outpainting() {
  return (
    <div className="outpainting-workarea">
      <OutpaintingPanel />
      <div className="outpainting-display" style={{ gridTemplateColumns: 'auto' }}>
        <OutpaintingImageDisplay />
      </div>
    </div>
  );
}
