import { CanvasSessionContextProvider } from 'features/controlLayers/components/SimpleSession/context';
import { StagingArea } from 'features/controlLayers/components/SimpleSession/StagingArea';
import { memo } from 'react';

export const SimpleSession = memo(({ id }: { id: string }) => {
  return (
    <CanvasSessionContextProvider type="simple" id={id}>
      <StagingArea />
    </CanvasSessionContextProvider>
  );
});
SimpleSession.displayName = 'SimpleSession';
