import { MenuItem } from '@chakra-ui/react';
import { FileButton } from '@mantine/core';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUpload } from 'react-icons/fa';

const UploadWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const loadWorkflowFromFile = useLoadWorkflowFromFile({ resetRef });
  return (
    <FileButton
      resetRef={resetRef}
      accept="application/json"
      onChange={loadWorkflowFromFile}
    >
      {(props) => (
        <MenuItem as="button" icon={<FaUpload />} {...props}>
          {t('workflows.uploadWorkflow')}
        </MenuItem>
      )}
    </FileButton>
  );
};

export default memo(UploadWorkflowMenuItem);
