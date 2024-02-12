import { Button, IconButton } from '@invoke-ai/ui-library';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

interface Props {
  full?: boolean;
  onSuccess?: () => void;
}

const UploadWorkflowMenuItem = ({ full, onSuccess }: Props) => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const loadWorkflowFromFile = useLoadWorkflowFromFile({ resetRef, onSuccess });

  const onDropAccepted = useCallback(
    (files: File[]) => {
      if (!files[0]) {
        return;
      }
      loadWorkflowFromFile(files[0]);
    },
    [loadWorkflowFromFile]
  );

  const { getRootProps } = useDropzone({
    accept: { 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });
  return (
    <>
      {full ? (
        <Button
          aria-label={t('workflows.uploadWorkflow')}
          tooltip={t('workflows.uploadWorkflow')}
          leftIcon={<PiUploadSimpleBold />}
          {...getRootProps()}
          pointerEvents="auto"
        >
          {t('workflows.uploadWorkflow')}
        </Button>
      ) : (
        <IconButton
          aria-label={t('workflows.uploadWorkflow')}
          tooltip={t('workflows.uploadWorkflow')}
          icon={<PiUploadSimpleBold />}
          {...getRootProps()}
          pointerEvents="auto"
        />
      )}
    </>
  );
};

export default memo(UploadWorkflowMenuItem);
