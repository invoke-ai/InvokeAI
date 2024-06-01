import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { preventDefault } from 'common/util/stopPropagation';
import { DROP_SHADOW } from 'features/gallery/components/ImageViewer/useImageViewer';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
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

export const ImageComparisonHover = memo(({ firstImage, secondImage }: Props) => {
  const { t } = useTranslation();
  const comparisonFit = useAppSelector((s) => s.gallery.comparisonFit);
  const [isMouseOver, setIsMouseOver] = useState(false);
  const onMouseOver = useCallback(() => {
    setIsMouseOver(true);
  }, []);
  const onMouseOut = useCallback(() => {
    setIsMouseOver(false);
  }, []);
  return (
    <Flex w="full" h="full" maxW="full" maxH="full" position="relative" alignItems="center" justifyContent="center">
      <Flex position="absolute" maxW="full" maxH="full" aspectRatio={firstImage.width / firstImage.height}>
        <Image
          id="image-comparison-first-image"
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
        >
          {t('gallery.viewerImage')}
        </Text>
        <Flex
          position="absolute"
          top={0}
          right={0}
          bottom={0}
          left={0}
          opacity={isMouseOver ? 1 : 0}
          transitionDuration="0.2s"
          transitionProperty="common"
        >
          <Image
            id="image-comparison-second-image"
            w={comparisonFit === 'fill' ? 'full' : secondImage.width}
            h={comparisonFit === 'fill' ? 'full' : secondImage.height}
            maxW={comparisonFit === 'contain' ? 'full' : undefined}
            maxH={comparisonFit === 'contain' ? 'full' : undefined}
            src={secondImage.image_url}
            fallbackSrc={secondImage.thumbnail_url}
            objectFit={comparisonFit}
            objectPosition="top left"
          />
          <Text
            position="absolute"
            bottom={4}
            insetInlineStart={4}
            textOverflow="clip"
            whiteSpace="nowrap"
            filter={DROP_SHADOW}
            color="base.50"
          >
            {t('gallery.compareImage')}
          </Text>
        </Flex>
        <Flex
          id="image-comparison-interaction-overlay"
          position="absolute"
          top={0}
          right={0}
          bottom={0}
          left={0}
          onMouseOver={onMouseOver}
          onMouseOut={onMouseOut}
          onContextMenu={preventDefault}
          userSelect="none"
        />
      </Flex>
    </Flex>
  );
});

ImageComparisonHover.displayName = 'ImageComparisonHover';
