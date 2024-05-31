import { ImgComparisonSlider } from '@img-comparison-slider/react';
import { Flex, Icon, Image, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { atom } from 'nanostores';
import { memo } from 'react';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';
import { useMeasure } from 'react-use';
import type { ImageDTO } from 'services/api/types';

const $compareWith = atom<ImageDTO | null>(null);

export const ImageSliderComparison = memo(() => {
  const [containerRef, containerDims] = useMeasure<HTMLDivElement>();
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const imageToCompare = useAppSelector((s) => s.gallery.selection[0]);
  // const imageToCompare = useStore($imageToCompare);
  const { imageA, imageB } = useAppSelector((s) => {
    const images = s.gallery.selection.slice(-2);
    return { imageA: images[0] ?? null, imageB: images[1] ?? null };
  });

  if (!imageA || !imageB) {
    return (
      <Flex w="full" h="full" maxW="full" maxH="full" alignItems="center" justifyContent="center" position="relative">
        <Text>Select images to compare</Text>
      </Flex>
    );
  }

  return (
    <Flex
      ref={containerRef}
      w="full"
      h="full"
      maxW="full"
      maxH="full"
      alignItems="center"
      justifyContent="center"
      position="relative"
    >
      <Flex top={0} right={0} bottom={0} left={0} position="absolute" alignItems="center" justifyContent="center">
        <ImgComparisonSlider>
          <Image
            slot="first"
            src={imageA.image_url}
            alt={imageA.image_name}
            w="full"
            h="full"
            maxW={containerDims.width}
            maxH={containerDims.height}
            backdropFilter="blur(20%)"
            objectPosition="top left"
            objectFit="contain"
          />
          <Image
            slot="second"
            src={imageB.image_url}
            alt={imageB.image_name}
            w="full"
            h="full"
            maxW={containerDims.width}
            maxH={containerDims.height}
            objectFit="contain"
            objectPosition="top left"
          />
          <Flex slot="handle" gap={4}>
            <Icon as={PiCaretLeftBold} />
            <Icon as={PiCaretRightBold} />
          </Flex>
        </ImgComparisonSlider>
      </Flex>
    </Flex>
  );
});

ImageSliderComparison.displayName = 'ImageSliderComparison';
