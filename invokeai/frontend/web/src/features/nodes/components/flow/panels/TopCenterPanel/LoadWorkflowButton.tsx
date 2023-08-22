import { FileButton } from '@mantine/core';
import IAIIconButton from 'common/components/IAIIconButton';
import { useLoadWorkflowFromFile } from 'features/nodes/hooks/useLoadWorkflowFromFile';
import { memo, useRef } from 'react';
import { FaUpload } from 'react-icons/fa';

const LoadWorkflowButton = () => {
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
          tooltip="Load Workflow"
          aria-label="Load Workflow"
          {...props}
        />
      )}
    </FileButton>
  );
};

export default memo(LoadWorkflowButton);
