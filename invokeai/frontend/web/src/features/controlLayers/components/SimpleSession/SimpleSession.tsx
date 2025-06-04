import type { CanvasSessionContextValue } from 'features/controlLayers/components/SimpleSession/context';
import {
  buildProgressDataAtom,
  CanvasSessionContextProvider,
} from 'features/controlLayers/components/SimpleSession/context';
import { StagingArea } from 'features/controlLayers/components/SimpleSession/StagingArea';
import type { SimpleSessionIdentifier } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useMemo } from 'react';

export const SimpleSession = memo(({ session }: { session: SimpleSessionIdentifier }) => {
  const ctx = useMemo(
    () => ({ session, $progressData: buildProgressDataAtom() }) satisfies CanvasSessionContextValue,
    [session]
  );

  return (
    <CanvasSessionContextProvider value={ctx}>
      <StagingArea />
    </CanvasSessionContextProvider>
  );
});
SimpleSession.displayName = 'SimpleSession';
