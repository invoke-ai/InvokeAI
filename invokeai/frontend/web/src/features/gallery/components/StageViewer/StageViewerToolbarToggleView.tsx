import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectStageViewerMode } from 'features/ui/store/uiSelectors';
import { stageViewerModeChanged } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { PiGridFour, PiListBold } from 'react-icons/pi';

export const StageViewerToggleView = memo(() => {
  const viewMode = useAppSelector(selectStageViewerMode);
  const dispatch = useAppDispatch();

  const onSelect = useCallback(
    (mode: 'grid' | 'linear') => {
      dispatch(stageViewerModeChanged(mode));
    },
    [dispatch]
  );

  return (
    <Flex borderWidth={1} borderRadius="base" p={1}>
      <IconButton
        size="sm"
        variant="ghost"
        colorScheme={viewMode === 'grid' ? 'blue' : 'base.500'}
        aria-label="grid"
        onClick={onSelect.bind(null, 'grid')}
        icon={<PiGridFour />}
      />
      <IconButton
        size="sm"
        variant="ghost"
        colorScheme={viewMode === 'linear' ? 'blue' : 'base.500'}
        aria-label="linear"
        onClick={onSelect.bind(null, 'linear')}
        icon={<PiListBold />}
      />
    </Flex>
  );
});

StageViewerToggleView.displayName = 'StageViewerToggleView';
