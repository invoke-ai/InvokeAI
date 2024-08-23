import { useAppSelector } from 'app/store/storeHooks';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const StagingAreaIsStagingGate = memo((props: PropsWithChildren) => {
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);

  if (!isStaging) {
    return null;
  }

  return props.children;
});

StagingAreaIsStagingGate.displayName = 'StagingAreaIsStagingGate';
