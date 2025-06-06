/* eslint-disable i18next/no-literal-string */

import { Flex, Heading, Icon, IconButton, Text } from '@invoke-ai/ui-library';
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
    <>
      <Icon as={PiTextAaBold} boxSize={8} color="base.500" />
      <Heading size="sm">Generate from Text</Heading>
      <Text color="base.300">Enter a prompt and Invoke.</Text>
      <Flex w="full" justifyContent="flex-end">
        <IconButton
          onClick={focusOnPrompt}
          aria-label="Focus on prompt"
          icon={<PiCursorTextBold />}
          variant="link"
          h={8}
        />
      </Flex>
    </>
  );
});
InitialStateGenerateFromText.displayName = 'InitialStateGenerateFromText';
