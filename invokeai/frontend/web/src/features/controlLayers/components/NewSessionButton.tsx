import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { sessionStarted } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';

export const NewSessionButton = memo(() => {
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(sessionStarted());
  }, [dispatch]);

  return <Button onClick={onClick}>New</Button>;
});

NewSessionButton.displayName = 'NewSessionButton';
