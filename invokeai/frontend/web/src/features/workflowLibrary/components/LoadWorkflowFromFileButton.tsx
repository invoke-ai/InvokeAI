import { FileButton } from '@mantine/core';
import IAIIconButton from 'common/components/IAIIconButton';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';

const UploadWorkflowButton = () => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const loadWorkflowFromFile = useLoadWorkflowFromFile(resetRef);
  return (
    <FileButton
      resetRef={resetRef}
      accept="application/json"
      onChange={loadWorkflowFromFile}
    >
      {(props) => (
        <IAIIconButton
          icon={<FaUpload />}
          tooltip={t('workflows.uploadWorkflow')}
          aria-label={t('workflows.uploadWorkflow')}
          {...props}
        />
      )}
    </FileButton>
  );
};

export default memo(UploadWorkflowButton);
