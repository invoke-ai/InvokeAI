import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { DROP_SHADOW } from 'features/gallery/components/ImageViewer/useImageViewer';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
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
  const { t } = useTranslation();
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
        <PanelGroup ref={panelGroupRef} direction="horizontal" id="image-comparison-side-by-side">
          <Panel minSize={20}>
            <Flex position="relative" w="full" h="full" alignItems="center" justifyContent="center">
              <Flex position="absolute" maxW="full" maxH="full" aspectRatio={firstImage.width / firstImage.height}>
                <Image
                  id="image-comparison-side-by-side-first-image"
                  w={firstImage.width}
                  h={firstImage.height}
                  maxW="full"
                  maxH="full"
                  src={firstImage.image_url}
                  fallbackSrc={firstImage.thumbnail_url}
                  objectFit="contain"
                />
                <Text
                  position="absolute"
                  bottom={4}
                  insetInlineStart={4}
                  textOverflow="clip"
                  whiteSpace="nowrap"
                  filter={DROP_SHADOW}
                  color="base.50"
                  userSelect="none"
                >
                  {t('gallery.viewerImage')}
                </Text>
              </Flex>
            </Flex>
          </Panel>
          <ResizeHandle
            id="image-comparison-side-by-side-handle"
            onDoubleClick={onDoubleClickHandle}
            orientation="vertical"
          />

          <Panel minSize={20}>
            <Flex position="relative" w="full" h="full" alignItems="center" justifyContent="center">
              <Flex position="absolute" maxW="full" maxH="full" aspectRatio={secondImage.width / secondImage.height}>
                <Image
                  id="image-comparison-side-by-side-first-image"
                  w={secondImage.width}
                  h={secondImage.height}
                  maxW="full"
                  maxH="full"
                  src={secondImage.image_url}
                  fallbackSrc={secondImage.thumbnail_url}
                  objectFit="contain"
                />
                <Text
                  position="absolute"
                  bottom={4}
                  insetInlineStart={4}
                  textOverflow="clip"
                  whiteSpace="nowrap"
                  filter={DROP_SHADOW}
                  color="base.50"
                  userSelect="none"
                >
                  {t('gallery.compareImage')}
                </Text>
              </Flex>
            </Flex>
          </Panel>
        </PanelGroup>
      </Flex>
    </Flex>
  );
});

ImageComparisonSideBySide.displayName = 'ImageComparisonSideBySide';
