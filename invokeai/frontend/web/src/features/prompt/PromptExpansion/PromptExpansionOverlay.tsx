import { Box, Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { usePromptExpansionTracking } from 'features/prompt/PromptExpansion/usePromptExpansionTracking';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagicWandBold } from 'react-icons/pi';

export const PromptExpansionOverlay = memo(() => {
  const { isPending } = usePromptExpansionTracking();
  const { t } = useTranslation();

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
      <Flex
        direction="column"
        alignItems="center"
        gap={3}
        color="invokeYellow.300"
      >
        <Box
          position="relative"
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <PiMagicWandBold size={24} />
          <Spinner
            size="sm"
            position="absolute"
            color="invokeYellow.400"
            thickness="2px"
          />
        </Box>
        <Text fontSize="sm" fontWeight="medium" textAlign="center">
          {t('prompt.expandingPrompt')}
        </Text>
      </Flex>
    </Box>
  );
});

PromptExpansionOverlay.displayName = 'PromptExpansionOverlay'; 