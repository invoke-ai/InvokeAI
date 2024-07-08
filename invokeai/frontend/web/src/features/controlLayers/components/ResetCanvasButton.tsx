import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { canvasReset } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { PiTrashBold } from 'react-icons/pi';

export const ResetCanvasButton = memo(() => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(canvasReset());
  }, [dispatch]);
  return <IconButton onClick={onClick} icon={<PiTrashBold />} aria-label="Reset canvas" colorScheme="error" />;
});

ResetCanvasButton.displayName = 'ResetCanvasButton';
