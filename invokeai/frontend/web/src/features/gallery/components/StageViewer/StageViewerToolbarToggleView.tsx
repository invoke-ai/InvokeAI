import { Divider, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectStageViewerMode } from 'features/ui/store/uiSelectors';
import { stageViewerModeChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGridFour, PiListBold } from 'react-icons/pi';

export const StageViewerToggleView = memo(() => {
  const { t } = useTranslation();
  const viewMode = useAppSelector(selectStageViewerMode);
  const dispatch = useAppDispatch();

  const onSelect = useCallback(
    (mode: 'grid' | 'linear') => {
      dispatch(stageViewerModeChanged(mode));
    },
    [dispatch]
  );

  return (
    <Flex>
      <IconButton
        icon={<PiGridFour />}
        size="sm"
        aria-label={t('stageViewer.gridView')}
        tooltip={t('stageViewer.gridView')}
        variant="link"
        colorScheme={viewMode === 'grid' ? 'blue' : 'base.500'}
        onClick={onSelect.bind(null, 'grid')}
        alignSelf="stretch"
      />

      <Divider orientation="vertical" h={8} mx={2} />

      <IconButton
        icon={<PiListBold />}
        size="sm"
        aria-label={t('stageViewer.linearView')}
        tooltip={t('stageViewer.linearView')}
        variant="link"
        colorScheme={viewMode === 'linear' ? 'blue' : 'base.500'}
        onClick={onSelect.bind(null, 'linear')}
        alignSelf="stretch"
      />
    </Flex>
  );
});

StageViewerToggleView.displayName = 'StageViewerToggleView';
