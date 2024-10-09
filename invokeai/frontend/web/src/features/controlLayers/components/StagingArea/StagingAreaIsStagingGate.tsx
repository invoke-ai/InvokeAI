import { useAppSelector } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';
import { useGetQueueCountsByDestinationQuery } from 'services/api/endpoints/queue';

// This hook just serves as a persistent subscriber for the queue count query.
const queueCountArg = { destination: 'canvas' };
const useCanvasQueueCountWatcher = () => {
  useGetQueueCountsByDestinationQuery(queueCountArg);
};

export const StagingAreaIsStagingGate = memo((props: PropsWithChildren) => {
  useCanvasQueueCountWatcher();
  const isStaging = useAppSelector(selectIsStaging);

  if (!isStaging) {
    return null;
  }

  return props.children;
});

StagingAreaIsStagingGate.displayName = 'StagingAreaIsStagingGate';
