import { CanvasSessionContextProvider } from 'features/controlLayers/components/SimpleSession/context';
import { InitialState } from 'features/controlLayers/components/SimpleSession/InitialState';
import { StagingArea } from 'features/controlLayers/components/SimpleSession/StagingArea';
import { memo } from 'react';

export const SimpleSession = memo(({ id }: { id: string | null }) => {
  if (id === null) {
    return <InitialState />;
  }
  return (
    <CanvasSessionContextProvider type="simple" id={id}>
      <StagingArea />
    </CanvasSessionContextProvider>
  );
});
SimpleSession.displayName = 'SimpleSession';
