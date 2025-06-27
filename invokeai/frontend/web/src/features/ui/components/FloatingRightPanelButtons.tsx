import { Flex, IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAutoLayoutContext } from 'features/ui/layouts/auto-layout-context';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImagesSquareBold } from 'react-icons/pi';

export const FloatingRightPanelButtons = memo(() => {
  return (
    <Flex pos="absolute" transform="translate(0, -50%)" minW={8} top="50%" insetInlineEnd={2}>
      <ToggleRightPanelButton />
    </Flex>
  );
});
FloatingRightPanelButtons.displayName = 'FloatingRightPanelButtons';

const ToggleRightPanelButton = memo(() => {
  const { t } = useTranslation();
  const { toggleRightPanel } = useAutoLayoutContext();

  return (
    <Tooltip label={t('accessibility.toggleRightPanel')} placement="start">
      <IconButton
        aria-label={t('accessibility.toggleRightPanel')}
        onClick={toggleRightPanel}
        icon={<PiImagesSquareBold />}
        h={48}
      />
    </Tooltip>
  );
});
ToggleRightPanelButton.displayName = 'ToggleRightPanelButton';
