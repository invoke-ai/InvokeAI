import { Grid, Image } from '@invoke-ai/ui-library';
import { useCallback, useEffect, useState } from 'react';
import type { ImageDTO } from 'services/api/types';

type SeamlessTextureViewerProps = {
  imageDTO: ImageDTO;
};

export default function SeamlessTextureViewer(props: SeamlessTextureViewerProps) {
  const [tileCount, setTileCount] = useState(150);
  const [gridWidth, setGridWidth] = useState(256);

  const tiles = Array.from({ length: tileCount }, (_, index) => index);

  const handleScroll = useCallback(
    (e: WheelEvent, tileCount: number, gridWidth: number, props: { imageDTO: ImageDTO }) => {
      e.preventDefault();
      const delta = e.deltaY;
      const zoomFactor = 0.25;

      const minTileCount = 10;
      const maxTileCount = 100;
      const newTileCount = Math.min(Math.max(tileCount + Math.sign(delta) * 10, minTileCount), maxTileCount);

      const minGridWidth = 100;
      const maxGridWidth = props.imageDTO.width;
      const newGridWidth = Math.min(
        Math.max(gridWidth - Math.sign(delta) * zoomFactor * gridWidth, minGridWidth),
        maxGridWidth
      );

      if (newGridWidth > props.imageDTO.width) {
        setTileCount(newTileCount);
      }
      setGridWidth(newGridWidth);
    },
    []
  );

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => handleScroll(e, tileCount, gridWidth, props);
    window.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      window.removeEventListener('wheel', handleWheel);
    };
  }, [handleScroll, tileCount, gridWidth, props]);

  return (
    <Grid width="100%" templateColumns={`repeat(auto-fill, minmax(${gridWidth}px, 1fr))`} position="relative">
      {tiles.map((tileIndex) => (
        <Image
          key={tileIndex}
          src={props.imageDTO.image_url}
          alt={`Tile ${tileIndex + 1}`}
          sx={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      ))}
    </Grid>
  );
}
