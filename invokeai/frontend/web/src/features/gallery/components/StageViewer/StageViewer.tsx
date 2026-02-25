import { Divider, Flex, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectStageViewerMode } from 'features/ui/store/uiSelectors';
import { memo } from 'react';

import { StageViewerGridView } from './StageViewerGridView';
import { StageViewerLinearView } from './StageViewerLinearView';
import { StageViewerToolbar } from './StageViewerToolbar';

const StageViewerSx: SystemStyleObject = {
  flexDir: 'column',
  w: 'full',
  h: 'full',
  overflow: 'hidden',
  gap: 2,
  position: 'relative',
};

export const StageViewer = memo(() => {
  const viewMode = useAppSelector(selectStageViewerMode);

  return (
    <Flex sx={StageViewerSx}>
      <StageViewerToolbar />
      <Divider />
      <Flex>
        {viewMode === 'grid' ? <StageViewerGridView /> : viewMode === 'linear' ? <StageViewerLinearView /> : null}
      </Flex>
    </Flex>
  );
});

StageViewer.displayName = 'StageViewer';
