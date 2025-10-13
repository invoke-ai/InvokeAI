import { Flex, Image } from '@invoke-ai/ui-library';
import type { ComparisonProps } from 'features/gallery/components/ImageViewer/common';
import { ImageComparisonLabel } from 'features/gallery/components/ImageViewer/ImageComparisonLabel';
import { VerticalResizeHandle } from 'features/ui/components/tabs/ResizeHandle';
import { memo, useCallback, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';
import type { ImageDTO } from 'services/api/types';

export const ImageComparisonSideBySide = memo(({ firstImage, secondImage }: ComparisonProps) => {
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const onDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }
    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  return (
    <Flex w="full" h="full" maxW="full" maxH="full" position="relative" alignItems="center" justifyContent="center">
      <Flex w="full" h="full" maxW="full" maxH="full" position="absolute" alignItems="center" justifyContent="center">
        <PanelGroup
          ref={panelGroupRef}
          direction="horizontal"
          id="image-comparison-side-by-side"
          autoSaveId="image-comparison-side-by-side"
        >
          <Panel minSize={20}>
            <SideBySideImage imageDTO={firstImage} type="first" />
          </Panel>
          <VerticalResizeHandle id="image-comparison-side-by-side-handle" onDoubleClick={onDoubleClickHandle} />
          <Panel minSize={20}>
            <SideBySideImage imageDTO={secondImage} type="second" />
          </Panel>
        </PanelGroup>
      </Flex>
    </Flex>
  );
});

ImageComparisonSideBySide.displayName = 'ImageComparisonSideBySide';

const SideBySideImage = memo(({ imageDTO, type }: { imageDTO: ImageDTO; type: 'first' | 'second' }) => {
  return (
    <Flex position="relative" w="full" h="full" alignItems="center" justifyContent="center">
      <Flex position="absolute" maxW="full" maxH="full" aspectRatio={imageDTO.width / imageDTO.height}>
        <Image
          id={`image-comparison-side-by-side-${type}-image`}
          w={imageDTO.width}
          h={imageDTO.height}
          maxW="full"
          maxH="full"
          src={imageDTO.image_url}
          fallbackSrc={imageDTO.thumbnail_url}
          objectFit="contain"
          borderRadius="base"
        />
        <ImageComparisonLabel type={type} />
      </Flex>
    </Flex>
  );
});
SideBySideImage.displayName = 'SideBySideImage';
