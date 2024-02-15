import { ButtonGroup, Flex, IconButton, Image } from '@invoke-ai/ui-library';
import { useFocusedMouseWheel } from 'features/showcase/hooks/useFocusedMouseWheel';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagnifyingGlassFill, PiMagnifyingGlassMinusBold, PiMagnifyingGlassPlusBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

interface LightBoxProps {
  imageDTO: ImageDTO;
}

const LightBox = (props: LightBoxProps) => {
  const { imageDTO } = props;
  const [zoomLevel, setZoomLevel] = useState(1);
  const zoomPercent = 1 + 20 / 100; // 20% -> Feels okay. Don't need to expose I think
  const { t } = useTranslation();
  const lightBoxRef = useRef<HTMLDivElement | null>(null);

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

  useFocusedMouseWheel(lightBoxRef, handleZoomScroll);

  return (
    <Flex sx={{ width: '100%', height: '100%', position: 'relative' }} ref={lightBoxRef}>
      <Image
        src={imageDTO.image_url}
        w="auto"
        h="auto"
        objectFit="contain"
        style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.3s ease' }}
      />
      <Flex sx={{ position: 'absolute', top: 2, left: '45%', gap: 2, margin: 'auto' }}>
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
      </Flex>
    </Flex>
  );
};

export default memo(LightBox);
