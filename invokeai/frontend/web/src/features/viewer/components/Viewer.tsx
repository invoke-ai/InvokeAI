import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ViewerDroppable } from 'features/viewer/components/ViewerDroppable';
import { ViewerImage } from 'features/viewer/components/ViewerImage';
import { ViewerInfo } from 'features/viewer/components/ViewerInfo';
import { ViewerProgress } from 'features/viewer/components/ViewerProgress';
import { ViewerToolbar } from 'features/viewer/components/ViewerToolbar';
import { memo } from 'react';

export const Viewer = memo(() => {
  const viewerMode = useAppSelector((s) => s.ui.viewerMode);

  return (
    <Flex position="relative" flexDirection="column" height="full" width="full" gap={4}>
      <ViewerToolbar />
      <Flex height="full" width="full" alignItems="center" justifyContent="center">
        {viewerMode === 'image' && <ViewerImage />}
        {viewerMode === 'info' && <ViewerInfo />}
        {viewerMode === 'progress' && <ViewerProgress />}
      </Flex>
      <ViewerDroppable />
    </Flex>
  );
});

Viewer.displayName = 'Viewer';
