import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { StagingAreaItemsList } from 'features/controlLayers/components/SimpleSession/StagingAreaItemsList';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo } from 'react';

export const StagingArea = memo(() => {
  const isStaging = useAppSelector(selectIsStaging);

  if (!isStaging) {
    return null;
  }

  return (
    <Flex position="absolute" flexDir="column" bottom={4} gap={2} align="center" justify="center" left={4} right={4}>
      <StagingAreaItemsList />
      <StagingAreaToolbar />
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';
