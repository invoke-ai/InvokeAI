import React from 'react';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import OutpaintingPanel from './OutpaintingPanel';

export default function Outpainting() {
  return (
    <div className="outpainting-workarea">
      <OutpaintingPanel />
      <div className="outpainting-display" style={{ gridTemplateColumns: 'auto' }}>
        <CurrentImageDisplay />
      </div>
    </div>
  );
}
