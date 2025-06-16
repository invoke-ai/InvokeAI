import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { InitialStateButtonGridItem } from 'features/controlLayers/components/SimpleSession/InitialStateButtonGridItem';
import { memo } from 'react';
import { PiCursorTextBold, PiTextAaBold } from 'react-icons/pi';

const focusOnPrompt = () => {
  const promptElement = document.getElementById('prompt');
  if (promptElement instanceof HTMLTextAreaElement) {
    promptElement.focus();
    promptElement.select();
  }
};

export const InitialStateGenerateFromText = memo(() => {
  return (
    <InitialStateButtonGridItem onClick={focusOnPrompt} position="relative" gap={8}>
      <Icon as={PiTextAaBold} boxSize={8} color="base.500" />
      <Flex flexDir="column" alignItems="flex-start" gap={2}>
        <Heading size="sm">Generate from Text</Heading>
        <Text color="base.300">Enter a prompt and Invoke.</Text>
      </Flex>
      <Flex position="absolute" right={3} bottom={3}>
        <PiCursorTextBold />
      </Flex>
    </InitialStateButtonGridItem>
  );
});
InitialStateGenerateFromText.displayName = 'InitialStateGenerateFromText';
