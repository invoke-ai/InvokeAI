import { Flex, IconButton, Tooltip } from '@invoke-ai/ui-library';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSidebarSimpleBold } from 'react-icons/pi';

export const FloatingLayerPanelButtons = memo(() => {
  return (
    <Flex pos="absolute" transform="translate(0, -50%)" minW={8} top="50%" insetInlineEnd={2}>
      <ToggleLayerPanelButton />
    </Flex>
  );
});
FloatingLayerPanelButtons.displayName = 'FloatingLayerPanelButtons';

const ToggleLayerPanelButton = memo(() => {
  const { t } = useTranslation();

  const handleClick = useCallback(() => {
    navigationApi.toggleRightPanel();
  }, []);

  return (
    <Tooltip label={t('accessibility.toggleRightPanel')} placement="start">
      <IconButton
        aria-label={t('accessibility.toggleRightPanel')}
        onClick={handleClick}
        icon={<PiSidebarSimpleBold />}
        h={48}
      />
    </Tooltip>
  );
});
ToggleLayerPanelButton.displayName = 'ToggleLayerPanelButton';
