/* eslint-disable i18next/no-literal-string */
import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { $simpleId } from 'features/ui/components/MainPanelContent';
import { memo, useCallback } from 'react';

export const StartOverButton = memo(() => {
  const dispatch = useAppDispatch();

  const startOver = useCallback(() => {
    // dispatch(canvasSessionTypeChanged({ type: 'simple' }));
    $simpleId.set(null);
  }, []);

  return (
    <Button size="sm" variant="link" alignSelf="stretch" onClick={startOver} px={2}>
      Start Over
    </Button>
  );
});
StartOverButton.displayName = 'StartOverButton';
