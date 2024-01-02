import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';

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
    <InvMenuItem as="button" icon={<FaUpload />} {...getRootProps()}>
      {t('workflows.uploadWorkflow')}
      <input {...getInputProps()} />
    </InvMenuItem>
  );
};

export default memo(UploadWorkflowMenuItem);
