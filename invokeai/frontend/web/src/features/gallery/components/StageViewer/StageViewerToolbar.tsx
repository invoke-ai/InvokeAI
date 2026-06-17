import { Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { BoardsDropdown } from './BoardsSwitcherDropdown';
import { StageViewerToggleView } from './StageViewerToolbarToggleView';

export const StageViewerToolbar = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex gap={4} alignItems="center" cursor="default" fontWeight="semibold" h={8} w="full" justifyContent="center" px={2}>
      <Text fontSize="sm">{t('common.board')}</Text>
      <BoardsDropdown />
      <Spacer />
      <StageViewerToggleView />
    </Flex>
  );
});

StageViewerToolbar.displayName = 'StageViewerToolbar';
