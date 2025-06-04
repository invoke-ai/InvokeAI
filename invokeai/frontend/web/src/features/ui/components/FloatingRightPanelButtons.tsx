import { Flex, IconButton, Tooltip } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesSquareBold } from 'react-icons/pi';

export const FloatingRightPanelButtons = memo((props: { onToggle: () => void }) => {
  return (
    <Flex pos="absolute" transform="translate(0, -50%)" minW={8} top="50%" insetInlineEnd={2}>
      <ToggleRightPanelButton onToggle={props.onToggle} />
    </Flex>
  );
});
FloatingRightPanelButtons.displayName = 'FloatingRightPanelButtons';

const ToggleRightPanelButton = memo((props: { onToggle: () => void }) => {
  const { t } = useTranslation();
  return (
    <Tooltip label={t('accessibility.toggleRightPanel')} placement="start">
      <IconButton
        aria-label={t('accessibility.toggleRightPanel')}
        onClick={props.onToggle}
        icon={<PiImagesSquareBold />}
        h={48}
      />
    </Tooltip>
  );
});
ToggleRightPanelButton.displayName = 'ToggleRightPanelButton';
