import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowMode, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import type { MouseEventHandler } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiPencilBold } from 'react-icons/pi';

export const ModeToggle = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectWorkflowMode);
  const { t } = useTranslation();

  const onClickEdit = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(workflowModeChanged('edit'));
    },
    [dispatch]
  );

  const onClickView = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      dispatch(workflowModeChanged('view'));
    },
    [dispatch]
  );

  return (
    <Flex justifyContent="flex-end">
      {mode === 'view' && (
        <IconButton
          aria-label={t('nodes.editMode')}
          tooltip={t('nodes.editMode')}
          onClick={onClickEdit}
          icon={<PiPencilBold />}
          variant="outline"
          size="sm"
        />
      )}
      {mode === 'edit' && (
        <IconButton
          aria-label={t('nodes.viewMode')}
          tooltip={t('nodes.viewMode')}
          onClick={onClickView}
          icon={<PiEyeBold />}
          variant="outline"
          size="sm"
        />
      )}
    </Flex>
  );
};
