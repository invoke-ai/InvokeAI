import { useAppSelector } from 'app/store/storeHooks';
import { AdvancedSession } from 'features/controlLayers/components/AdvancedSession/AdvancedSession';
import { NoSession } from 'features/controlLayers/components/NoSession/NoSession';
import { SimpleSession } from 'features/controlLayers/components/SimpleSession/SimpleSession';
import { selectCanvasSessionId, selectCanvasSessionType } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const CanvasMainPanelContent = memo(() => {
  const type = useAppSelector(selectCanvasSessionType);
  const id = useAppSelector(selectCanvasSessionId);

  if (type === 'simple') {
    if (id === null) {
      return <NoSession />;
    } else {
      return <SimpleSession id={id} />;
    }
  }

  if (type === 'advanced') {
    return <AdvancedSession id={id} />;
  }

  assert<Equals<never, typeof type>>(false, 'Unexpected session type');
});
CanvasMainPanelContent.displayName = 'CanvasMainPanelContent';
