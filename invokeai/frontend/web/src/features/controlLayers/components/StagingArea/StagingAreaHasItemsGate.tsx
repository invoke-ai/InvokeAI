import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const StagingAreaHasItemsGate = memo((props: PropsWithChildren) => {
  const ctx = useCanvasSessionContext();
  const hasItems = useStore(ctx.$hasItems);

  if (!hasItems) {
    return null;
  }
  return props.children;
});

StagingAreaHasItemsGate.displayName = 'StagingAreaHasItemsGate';
