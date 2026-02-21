import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, IconButton } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { useNextPrevItemNavigation } from './useNextPrevItemNavigation';

const ARROW_SIZE = 48;

const NextPrevItemButtons = ({ inset = 8 }: { inset?: ChakraProps['insetInlineStart' | 'insetInlineEnd'] }) => {
  const { t } = useTranslation();
  const { goToPreviousImage, goToNextImage, isOnFirstItem, isOnLastItem, isFetching } = useNextPrevItemNavigation();

  return (
    <Box pos="relative" h="full" w="full">
      {!isOnFirstItem && (
        <IconButton
          position="absolute"
          top="50%"
          transform="translate(0, -50%)"
          aria-label={t('accessibility.previousImage')}
          icon={<PiCaretLeftBold size={ARROW_SIZE} />}
          variant="unstyled"
          padding={0}
          minW={0}
          minH={0}
          w={`${ARROW_SIZE}px`}
          h={`${ARROW_SIZE}px`}
          onClick={goToPreviousImage}
          isDisabled={isFetching}
          color="base.100"
          pointerEvents="auto"
          insetInlineStart={inset}
        />
      )}
      {!isOnLastItem && (
        <IconButton
          position="absolute"
          top="50%"
          transform="translate(0, -50%)"
          aria-label={t('accessibility.nextImage')}
          icon={<PiCaretRightBold size={ARROW_SIZE} />}
          variant="unstyled"
          padding={0}
          minW={0}
          minH={0}
          w={`${ARROW_SIZE}px`}
          h={`${ARROW_SIZE}px`}
          onClick={goToNextImage}
          isDisabled={isFetching}
          color="base.100"
          pointerEvents="auto"
          insetInlineEnd={inset}
        />
      )}
    </Box>
  );
};

export default memo(NextPrevItemButtons);
