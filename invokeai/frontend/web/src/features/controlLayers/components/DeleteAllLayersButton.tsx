import { Button } from '@invoke-ai/ui-library';
import { allLayersDeleted } from 'app/store/middleware/listenerMiddleware/listeners/regionalControlToControlAdapterBridge';
import { useAppDispatch } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

export const DeleteAllLayersButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(allLayersDeleted());
  }, [dispatch]);

  return (
    <Button onClick={onClick} leftIcon={<PiTrashSimpleBold />} variant="ghost" colorScheme="error">
      {t('regionalPrompts.deleteAll')}
    </Button>
  );
});

DeleteAllLayersButton.displayName = 'DeleteAllLayersButton';
