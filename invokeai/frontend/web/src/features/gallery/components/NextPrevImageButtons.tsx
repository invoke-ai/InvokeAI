import { ChakraProps, Flex, Grid, IconButton, Spinner } from '@chakra-ui/react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaAngleDoubleRight, FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import { useNextPrevImage } from '../hooks/useNextPrevImage';

const nextPrevButtonTriggerAreaStyles: ChakraProps['sx'] = {
  height: '100%',
  width: '15%',
  alignItems: 'center',
  pointerEvents: 'auto',
};
const nextPrevButtonStyles: ChakraProps['sx'] = {
  color: 'base.100',
};

const NextPrevImageButtons = () => {
  const { t } = useTranslation();

  const {
    handlePrevImage,
    handleNextImage,
    isOnFirstImage,
    isOnLastImage,
    handleLoadMoreImages,
    areMoreImagesAvailable,
    isFetching,
  } = useNextPrevImage();

  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const handleCurrentImagePreviewMouseOver = useCallback(() => {
    setShouldShowNextPrevButtons(true);
  }, []);

  const handleCurrentImagePreviewMouseOut = useCallback(() => {
    setShouldShowNextPrevButtons(false);
  }, []);

  return (
    <Flex
      sx={{
        justifyContent: 'space-between',
        height: '100%',
        width: '100%',
        pointerEvents: 'none',
      }}
    >
      <Grid
        sx={{
          ...nextPrevButtonTriggerAreaStyles,
          justifyContent: 'flex-start',
        }}
        onMouseOver={handleCurrentImagePreviewMouseOver}
        onMouseOut={handleCurrentImagePreviewMouseOut}
      >
        {shouldShowNextPrevButtons && !isOnFirstImage && (
          <IconButton
            aria-label={t('accessibility.previousImage')}
            icon={<FaAngleLeft size={64} />}
            variant="unstyled"
            onClick={handlePrevImage}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
      </Grid>
      <Grid
        sx={{
          ...nextPrevButtonTriggerAreaStyles,
          justifyContent: 'flex-end',
        }}
        onMouseOver={handleCurrentImagePreviewMouseOver}
        onMouseOut={handleCurrentImagePreviewMouseOut}
      >
        {shouldShowNextPrevButtons && !isOnLastImage && (
          <IconButton
            aria-label={t('accessibility.nextImage')}
            icon={<FaAngleRight size={64} />}
            variant="unstyled"
            onClick={handleNextImage}
            boxSize={16}
            sx={nextPrevButtonStyles}
          />
        )}
        {shouldShowNextPrevButtons &&
          isOnLastImage &&
          areMoreImagesAvailable &&
          !isFetching && (
            <IconButton
              aria-label={t('accessibility.loadMore')}
              icon={<FaAngleDoubleRight size={64} />}
              variant="unstyled"
              onClick={handleLoadMoreImages}
              boxSize={16}
              sx={nextPrevButtonStyles}
            />
          )}
        {shouldShowNextPrevButtons &&
          isOnLastImage &&
          areMoreImagesAvailable &&
          isFetching && (
            <Flex
              sx={{
                w: 16,
                h: 16,
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Spinner opacity={0.5} size="xl" />
            </Flex>
          )}
      </Grid>
    </Flex>
  );
};

export default memo(NextPrevImageButtons);
