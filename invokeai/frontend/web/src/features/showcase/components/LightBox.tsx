import { ButtonGroup, Flex, IconButton, Image } from '@invoke-ai/ui-library';
import { useFocusedMouseWheel } from 'features/showcase/hooks/useFocusedMouseWheel';
import { useMousePan } from 'features/showcase/hooks/useMousePan';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsIn,
  PiMagnifyingGlassFill,
  PiMagnifyingGlassMinusBold,
  PiMagnifyingGlassPlusBold,
  PiScan,
} from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

interface LightBoxProps {
  imageDTO: ImageDTO;
}

const LightBox = (props: LightBoxProps) => {
  const { imageDTO } = props;
  const { t } = useTranslation();

  // Refs
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Zoom
  const [zoomLevel, setZoomLevel] = useState(1);
  const zoomPercent = 1 + 20 / 100; // 20% -> Feels okay. Don't need to expose I think

  // Pan
  const { panPosition, handleMousePanDown, handleMousePanUp, handleMousePanMove, resetPan } = useMousePan(imageRef);

  const handleZoomIn = useCallback(() => {
    setZoomLevel(zoomLevel * zoomPercent);
  }, [zoomLevel, zoomPercent]);

  const handleZoomOut = useCallback(() => {
    setZoomLevel(zoomLevel / zoomPercent);
  }, [zoomLevel, zoomPercent]);

  const resetZoom = useCallback(() => {
    setZoomLevel(1);
  }, [setZoomLevel]);

  const handleZoomScroll = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1 / zoomPercent : zoomPercent;
      setZoomLevel(zoomLevel * zoomFactor);
    },
    [zoomLevel, zoomPercent]
  );

  useFocusedMouseWheel(imageRef, handleZoomScroll);

  const resetView = useCallback(() => {
    resetZoom();
    resetPan();
  }, [resetPan, resetZoom]);

  return (
    <Flex sx={{ width: '100%', height: '100%', position: 'relative' }} onMouseLeave={handleMousePanUp}>
      <Image
        ref={imageRef}
        src={imageDTO.image_url}
        w="100%"
        h="auto"
        objectFit="contain"
        style={{
          scale: `${zoomLevel}`,
          translate: `${panPosition.x}px ${panPosition.y}px`,
          transition: 'scale 0.1s ease-out',
          cursor: 'move',
        }}
        onMouseDown={handleMousePanDown}
        onMouseUp={handleMousePanUp}
        onMouseMove={handleMousePanMove}
        onMouseLeave={handleMousePanUp}
        onDoubleClick={resetView}
      />
      <Flex sx={{ position: 'absolute', top: 2, w: 'full', h: 'max-content', gap: 2, justifyContent: 'center' }}>
        <ButtonGroup>
          <IconButton
            icon={<PiScan />}
            tooltip={`${t('showcase.resetView')}`}
            aria-label={`${t('showcase.resetView')}`}
            onClick={resetView}
          />
        </ButtonGroup>
        <ButtonGroup>
          <IconButton
            icon={<PiMagnifyingGlassPlusBold />}
            tooltip={`${t('showcase.zoomIn')}`}
            aria-label={`${t('showcase.zoomIn')}`}
            onClick={handleZoomIn}
          />
          <IconButton
            icon={<PiMagnifyingGlassFill />}
            tooltip={`${t('showcase.resetZoom')}`}
            aria-label={`${t('showcase.resetZoom')}`}
            onClick={resetZoom}
          />
          <IconButton
            icon={<PiMagnifyingGlassMinusBold />}
            tooltip={`${t('showcase.zoomOut')}`}
            aria-label={`${t('showcase.zoomOut')}`}
            onClick={handleZoomOut}
          />
        </ButtonGroup>
        <ButtonGroup>
          <IconButton
            icon={<PiArrowsIn />}
            tooltip={`${t('showcase.resetPosition')}`}
            aria-label={`${t('showcase.resetPosition')}`}
            onClick={resetPan}
          />
        </ButtonGroup>
      </Flex>
    </Flex>
  );
};

export default memo(LightBox);
