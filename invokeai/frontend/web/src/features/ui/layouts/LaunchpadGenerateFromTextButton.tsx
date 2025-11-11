import { Flex, Heading, Icon, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useActiveCanvasIsStaging } from 'features/controlLayers/hooks/useCanvasIsStaging';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import { LaunchpadButton } from 'features/ui/layouts/LaunchpadButton';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCursorTextBold, PiTextAaBold } from 'react-icons/pi';

const focusOnPrompt = () => {
  const promptElement = document.querySelector('.positive-prompt-textarea');
  if (promptElement instanceof HTMLTextAreaElement) {
    promptElement.focus();
    promptElement.select();
  }
};

export const LaunchpadGenerateFromTextButton = memo((props: { extraAction?: () => void }) => {
  const { t } = useTranslation();
  const tab = useAppSelector(selectActiveTab);
  const isStaging = useActiveCanvasIsStaging();

  const onClick = useCallback(() => {
    focusOnPrompt();
    props.extraAction?.();
  }, [props]);
  return (
    <LaunchpadButton onClick={onClick} position="relative" gap={8} isDisabled={tab === 'canvas' && isStaging}>
      <Icon as={PiTextAaBold} boxSize={8} color="base.500" />
      <Flex flexDir="column" alignItems="flex-start" gap={2}>
        <Heading size="sm">{t('ui.launchpad.generateFromText.title')}</Heading>
        <Text>{t('ui.launchpad.generateFromText.description')}</Text>
      </Flex>
      <Flex position="absolute" right={3} bottom={3}>
        <PiCursorTextBold />
      </Flex>
    </LaunchpadButton>
  );
});
LaunchpadGenerateFromTextButton.displayName = 'LaunchpadGenerateFromTextButton';
