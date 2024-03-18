import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilBold } from 'react-icons/pi';

export const ModeToggle = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector((s) => s.workflow.mode);
  const { t } = useTranslation();

  const onClickEdit = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  const onClickView = useCallback(() => {
    dispatch(workflowModeChanged('view'));
  }, [dispatch]);

  return (
    <Flex justifyContent="flex-end">
      {mode === 'view' && (
        <IconButton
          aria-label={t('nodes.editMode')}
          tooltip={t('nodes.editMode')}
          onClick={onClickEdit}
          icon={<PiPencilBold />}
          colorScheme="invokeBlue"
        />
      )}
      {mode === 'edit' && (
        <IconButton
          aria-label={t('nodes.viewMode')}
          tooltip={t('nodes.viewMode')}
          onClick={onClickView}
          icon={<PiEyeBold />}
          colorScheme="invokeBlue"
        />
      )}
    </Flex>
  );
};
