import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { LaunchpadButton } from 'features/controlLayers/components/SimpleSession/LaunchpadButton';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { memo, useCallback } from 'react';
import { PiCursorTextBold, PiTextAaBold } from 'react-icons/pi';

const focusOnPrompt = (el: HTMLElement) => {
  const promptElement = el.querySelector('.positive-prompt-textarea');
  if (promptElement instanceof HTMLTextAreaElement) {
    promptElement.focus();
    promptElement.select();
  }
};

export const LaunchpadGenerateFromTextButton = memo((props: { extraAction?: () => void }) => {
  const { rootRef } = useAutoLayoutContext();
  const onClick = useCallback(() => {
    const el = rootRef.current;
    if (!el) {
      return;
    }
    focusOnPrompt(el);
    props.extraAction?.();
  }, [props, rootRef]);
  return (
    <LaunchpadButton onClick={onClick} position="relative" gap={8}>
      <Icon as={PiTextAaBold} boxSize={8} color="base.500" />
      <Flex flexDir="column" alignItems="flex-start" gap={2}>
        <Heading size="sm">Generate from Text</Heading>
        <Text color="base.300">Enter a prompt and Invoke.</Text>
      </Flex>
      <Flex position="absolute" right={3} bottom={3}>
        <PiCursorTextBold />
      </Flex>
    </LaunchpadButton>
  );
});
LaunchpadGenerateFromTextButton.displayName = 'LaunchpadGenerateFromTextButton';
