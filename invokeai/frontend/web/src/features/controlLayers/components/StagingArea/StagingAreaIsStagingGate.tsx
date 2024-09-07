import { useAppSelector } from 'app/store/storeHooks';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const StagingAreaIsStagingGate = memo((props: PropsWithChildren) => {
  const isStaging = useAppSelector(selectIsStaging);

  if (!isStaging) {
    return null;
  }

  return props.children;
});

StagingAreaIsStagingGate.displayName = 'StagingAreaIsStagingGate';
