import { useAppSelector } from 'app/store/storeHooks';
import { AdvancedSession } from 'features/controlLayers/components/AdvancedSession/AdvancedSession';
import { NoSession } from 'features/controlLayers/components/NoSession/NoSession';
import { SimpleSession } from 'features/controlLayers/components/SimpleSession/SimpleSession';
import { selectCanvasSession } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const CanvasMainPanelContent = memo(() => {
  const session = useAppSelector(selectCanvasSession);

  if (session === null) {
    return <NoSession />;
  }

  if (session.type === 'simple') {
    return <SimpleSession session={session} />;
  }

  if (session.type === 'advanced') {
    return <AdvancedSession session={session} />;
  }

  assert<Equals<never, typeof session>>(false, 'Unexpected session');
});
CanvasMainPanelContent.displayName = 'CanvasMainPanelContent';
