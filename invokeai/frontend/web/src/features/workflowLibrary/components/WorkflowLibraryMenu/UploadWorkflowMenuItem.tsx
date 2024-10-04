import { MenuItem } from '@invoke-ai/ui-library';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

const UploadWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const loadWorkflowFromFile = useLoadWorkflowFromFile({ resetRef });

  const onDropAccepted = useCallback(
    (files: File[]) => {
      if (!files[0]) {
        return;
      }
      loadWorkflowFromFile(files[0]);
    },
    [loadWorkflowFromFile]
  );

  const { getRootProps, getInputProps } = useDropzone({
    accept: { 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });
  return (
    <MenuItem as="button" icon={<PiUploadSimpleBold />} {...getRootProps()}>
      {t('workflows.uploadWorkflow')}
      <input {...getInputProps()} />
    </MenuItem>
  );
};

export default memo(UploadWorkflowMenuItem);
