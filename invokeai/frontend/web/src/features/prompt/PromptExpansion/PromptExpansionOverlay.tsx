import { Box, Flex, Image, Spinner, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { PromptExpansionResultOverlay } from 'features/prompt/PromptExpansion/PromptExpansionResultOverlay';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagicWandBold } from 'react-icons/pi';

import { promptExpansionApi } from './state';

export const PromptExpansionOverlay = memo(() => {
  const { isSuccess, isPending, result, imageDTO } = useStore(promptExpansionApi.$state);
  const { t } = useTranslation();

  // Show result overlay when completed
  if (isSuccess) {
    return <PromptExpansionResultOverlay expandedText={result} />;
  }

  // Show pending overlay when pending
  if (!isPending) {
    return null;
  }

  return (
    <Box
      position="absolute"
      top={0}
      left={0}
      right={0}
      bottom={0}
      bg="base.900"
      opacity={0.8}
      borderRadius="base"
      zIndex={10}
      display="flex"
      alignItems="center"
      justifyContent="center"
      animation="pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite"
    >
      {/* Show dimmed source image if available */}
      {imageDTO && (
        <Box
          position="absolute"
          top={2}
          left={2}
          right={2}
          bottom={2}
          opacity={0.5}
          borderRadius="base"
          overflow="hidden"
        >
          <Image src={imageDTO.thumbnail_url} objectFit="contain" w="full" h="full" borderRadius="base" />
        </Box>
      )}

      <Flex direction="column" alignItems="center" gap={3} color="invokeYellow.300" position="relative" zIndex={1}>
        <Box position="relative" display="flex" alignItems="center" justifyContent="center">
          <PiMagicWandBold size={24} />
          <Spinner size="sm" position="absolute" color="invokeYellow.400" thickness="2px" />
        </Box>
        <Text fontSize="sm" fontWeight="medium" textAlign="center">
          {t('prompt.expandingPrompt')}
        </Text>
      </Flex>
    </Box>
  );
});

PromptExpansionOverlay.displayName = 'PromptExpansionOverlay';
