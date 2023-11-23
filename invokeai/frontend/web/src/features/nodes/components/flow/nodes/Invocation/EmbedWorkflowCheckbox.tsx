import { Checkbox, Flex, FormControl, FormLabel } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEmbedWorkflow } from 'features/nodes/hooks/useEmbedWorkflow';
import { useWithWorkflow } from 'features/nodes/hooks/useWithWorkflow';
import { nodeEmbedWorkflowChanged } from 'features/nodes/store/nodesSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const EmbedWorkflowCheckbox = ({ nodeId }: { nodeId: string }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const withWorkflow = useWithWorkflow(nodeId);
  const embedWorkflow = useEmbedWorkflow(nodeId);
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        nodeEmbedWorkflowChanged({
          nodeId,
          embedWorkflow: e.target.checked,
        })
      );
    },
    [dispatch, nodeId]
  );

  if (!withWorkflow) {
    return null;
  }

  return (
    <FormControl as={Flex} sx={{ alignItems: 'center', gap: 2, w: 'auto' }}>
      <FormLabel sx={{ fontSize: 'xs', mb: '1px' }}>
        {t('metadata.workflow')}
      </FormLabel>
      <Checkbox
        className="nopan"
        size="sm"
        onChange={handleChange}
        isChecked={embedWorkflow}
      />
    </FormControl>
  );
};

export default memo(EmbedWorkflowCheckbox);
