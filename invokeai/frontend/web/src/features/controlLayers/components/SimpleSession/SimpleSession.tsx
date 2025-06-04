import { CanvasSessionContextProvider } from 'features/controlLayers/components/SimpleSession/context';
import { StagingArea } from 'features/controlLayers/components/SimpleSession/StagingArea';
import type { SimpleSessionIdentifier } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo } from 'react';

export const SimpleSession = memo(({ session }: { session: SimpleSessionIdentifier }) => {
  return (
    <CanvasSessionContextProvider session={session}>
      <StagingArea />
    </CanvasSessionContextProvider>
  );
});
SimpleSession.displayName = 'SimpleSession';
