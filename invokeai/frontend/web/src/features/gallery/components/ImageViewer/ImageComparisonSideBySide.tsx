import { Flex } from '@invoke-ai/ui-library';
import IAIDndImage from 'common/components/IAIDndImage';
import type { ImageDraggableData } from 'features/dnd/types';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo, useCallback, useMemo, useRef } from 'react';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';
import type { ImageDTO } from 'services/api/types';

type Props = {
  /**
   * The first image to compare
   */
  firstImage: ImageDTO;
  /**
   * The second image to compare
   */
  secondImage: ImageDTO;
};

export const ImageComparisonSideBySide = memo(({ firstImage, secondImage }: Props) => {
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const onDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }
    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  const firstImageDraggableData = useMemo<ImageDraggableData>(
    () => ({
      id: 'image-compare-first-image',
      payloadType: 'IMAGE_DTO',
      payload: { imageDTO: firstImage },
    }),
    [firstImage]
  );

  const secondImageDraggableData = useMemo<ImageDraggableData>(
    () => ({
      id: 'image-compare-second-image',
      payloadType: 'IMAGE_DTO',
      payload: { imageDTO: secondImage },
    }),
    [secondImage]
  );

  return (
    <Flex w="full" h="full" maxW="full" maxH="full" position="relative" alignItems="center" justifyContent="center">
      <Flex w="full" h="full" maxW="full" maxH="full" position="absolute" alignItems="center" justifyContent="center">
        <PanelGroup ref={panelGroupRef} direction="horizontal" id="image-comparison-side-by-side">
          <Panel minSize={20}>
            <Flex w="full" h="full" alignItems="center" justifyContent="center">
              <IAIDndImage
                imageDTO={firstImage}
                isDropDisabled={true}
                draggableData={firstImageDraggableData}
                useThumbailFallback
              />
            </Flex>
          </Panel>
          <ResizeHandle
            id="image-comparison-side-by-side-handle"
            onDoubleClick={onDoubleClickHandle}
            orientation="vertical"
          />

          <Panel minSize={20}>
            <Flex w="full" h="full" alignItems="center" justifyContent="center">
              <IAIDndImage
                imageDTO={secondImage}
                isDropDisabled={true}
                draggableData={secondImageDraggableData}
                useThumbailFallback
              />
            </Flex>
          </Panel>
        </PanelGroup>
      </Flex>
    </Flex>
  );
});

ImageComparisonSideBySide.displayName = 'ImageComparisonSideBySide';
