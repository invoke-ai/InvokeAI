import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowMode, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilBold } from 'react-icons/pi';

export const ModeToggle = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectWorkflowMode);
  const { t } = useTranslation();

  const onPointerUpEdit = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  const onPointerUpView = useCallback(() => {
    dispatch(workflowModeChanged('view'));
  }, [dispatch]);

  return (
    <Flex justifyContent="flex-end">
      {mode === 'view' && (
        <IconButton
          aria-label={t('nodes.editMode')}
          tooltip={t('nodes.editMode')}
          onPointerUp={onPointerUpEdit}
          icon={<PiPencilBold />}
          colorScheme="invokeBlue"
        />
      )}
      {mode === 'edit' && (
        <IconButton
          aria-label={t('nodes.viewMode')}
          tooltip={t('nodes.viewMode')}
          onPointerUp={onPointerUpView}
          icon={<PiEyeBold />}
          colorScheme="invokeBlue"
        />
      )}
    </Flex>
  );
};
