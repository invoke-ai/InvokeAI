import { FormControl, FormLabel, Tooltip } from '@invoke-ai/ui-library';
import { FieldHandle } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldHandle';
import { OutputFieldWrapper } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldWrapper';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldOutputTemplate } from 'features/nodes/hooks/useFieldOutputTemplate';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

import FieldTooltipContent from './FieldTooltipContent';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const OutputFieldNodesEditorView = memo(({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useFieldOutputTemplate(nodeId, fieldName);

  const { isConnected, isConnectionInProgress, isConnectionStartField, validationResult, shouldDim } =
    useConnectionState(nodeId, fieldName, 'outputs');

  return (
    <OutputFieldWrapper shouldDim={shouldDim}>
      <Tooltip
        label={<FieldTooltipContent nodeId={nodeId} fieldName={fieldName} kind="outputs" />}
        openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
        placement="top"
        shouldWrapChildren
      >
        <FormControl isDisabled={isConnected}>
          <FormLabel mb={0}>{fieldTemplate?.title}</FormLabel>
        </FormControl>
      </Tooltip>
      <FieldHandle
        handleType="source"
        fieldTemplate={fieldTemplate}
        isConnectionInProgress={isConnectionInProgress}
        isConnectionStartField={isConnectionStartField}
        validationResult={validationResult}
      />
    </OutputFieldWrapper>
  );
});

OutputFieldNodesEditorView.displayName = 'OutputFieldNodesEditorView';
