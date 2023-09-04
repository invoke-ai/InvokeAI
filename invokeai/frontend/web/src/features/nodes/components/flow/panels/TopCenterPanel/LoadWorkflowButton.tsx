import { FileButton } from '@mantine/core';
import IAIIconButton from 'common/components/IAIIconButton';
import { useLoadWorkflowFromFile } from 'features/nodes/hooks/useLoadWorkflowFromFile';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';

const LoadWorkflowButton = () => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const loadWorkflowFromFile = useLoadWorkflowFromFile();
  return (
    <FileButton
      resetRef={resetRef}
      accept="application/json"
      onChange={loadWorkflowFromFile}
    >
      {(props) => (
        <IAIIconButton
          icon={<FaUpload />}
          tooltip={t('nodes.loadWorkflow')}
          aria-label={t('nodes.loadWorkflow')}
          {...props}
        />
      )}
    </FileButton>
  );
};

export default memo(LoadWorkflowButton);
