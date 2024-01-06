import type { ChakraProps } from '@chakra-ui/react';
import { Box, Flex, Spinner } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaAngleDoubleRight, FaAngleLeft, FaAngleRight } from 'react-icons/fa';

const nextPrevButtonStyles: ChakraProps['sx'] = {
  color: 'base.100',
  pointerEvents: 'auto',
};

const NextPrevImageButtons = () => {
  const { t } = useTranslation();

  const { handleLeftImage, handleRightImage, isOnFirstImage, isOnLastImage } =
    useGalleryNavigation();

  const {
    areMoreImagesAvailable,
    handleLoadMoreImages,
    queryResult: { isFetching },
  } = useGalleryImages();

  return (
    <Box pos="relative" h="full" w="full">
      <Box
        pos="absolute"
        top="50%"
        transform="translate(0, -50%)"
        insetInlineStart={0}
      >
        {!isOnFirstImage && (
          <InvIconButton
            aria-label={t('accessibility.previousImage')}
            icon={<FaAngleLeft size={64} />}
            variant="unstyled"
            onClick={handleLeftImage}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
      </Box>
      <Box
        pos="absolute"
        top="50%"
        transform="translate(0, -50%)"
        insetInlineEnd={0}
      >
        {!isOnLastImage && (
          <InvIconButton
            aria-label={t('accessibility.nextImage')}
            icon={<FaAngleRight size={64} />}
            variant="unstyled"
            onClick={handleRightImage}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
        {isOnLastImage && areMoreImagesAvailable && !isFetching && (
          <InvIconButton
            aria-label={t('accessibility.loadMore')}
            icon={<FaAngleDoubleRight size={64} />}
            variant="unstyled"
            onClick={handleLoadMoreImages}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
        {isOnLastImage && areMoreImagesAvailable && isFetching && (
          <Flex w={16} h={16} alignItems="center" justifyContent="center">
            <Spinner opacity={0.5} size="xl" />
          </Flex>
        )}
      </Box>
    </Box>
  );
};

export default memo(NextPrevImageButtons);
