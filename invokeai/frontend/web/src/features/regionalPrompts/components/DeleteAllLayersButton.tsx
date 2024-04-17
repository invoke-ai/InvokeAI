import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { allLayersDeleted } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';

export const DeleteAllLayersButton = memo(() => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(allLayersDeleted());
  }, [dispatch]);

  return <Button onClick={onClick}>Delete All</Button>;
});

DeleteAllLayersButton.displayName = 'DeleteAllLayersButton';
