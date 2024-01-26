import {
  ConfirmationAlertDialog,
  FormControl,
  FormLabel,
  Input,
  MenuItem,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useSaveWorkflowAs } from 'features/workflowLibrary/hooks/useSaveWorkflowAs';
import { getWorkflowCopyName } from 'features/workflowLibrary/util/getWorkflowCopyName';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

const SaveWorkflowAsButton = () => {
  const currentName = useAppSelector((s) => s.workflow.name);
  const { t } = useTranslation();
  const { saveWorkflowAs } = useSaveWorkflowAs();
  const [name, setName] = useState(getWorkflowCopyName(currentName));
  const { isOpen, onOpen, onClose } = useDisclosure();
  const inputRef = useRef<HTMLInputElement>(null);

  const onOpenCallback = useCallback(() => {
    setName(getWorkflowCopyName(currentName));
    onOpen();
    inputRef.current?.focus();
  }, [currentName, onOpen]);

  const onSave = useCallback(async () => {
    saveWorkflowAs({ name, onSuccess: onClose, onError: onClose });
  }, [name, onClose, saveWorkflowAs]);

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  }, []);

  return (
    <>
      <MenuItem as="button" icon={<PiCopyBold />} onClick={onOpenCallback}>
        {t('workflows.saveWorkflowAs')}
      </MenuItem>

      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('workflows.saveWorkflowAs')}
        acceptCallback={onSave}
      >
        <FormControl>
          <FormLabel>{t('workflows.workflowName')}</FormLabel>
          <Input
            ref={inputRef}
            value={name}
            onChange={onChange}
            placeholder={t('workflows.workflowName')}
          />
        </FormControl>
      </ConfirmationAlertDialog>
    </>
  );
};

export default memo(SaveWorkflowAsButton);
