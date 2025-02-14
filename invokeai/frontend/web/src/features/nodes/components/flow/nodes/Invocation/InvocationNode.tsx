import { Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import NodeWrapper from 'features/nodes/components/flow/nodes/common/NodeWrapper';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { OutputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldGate';
import { OutputFieldNodesEditorView } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldNodesEditorView';
import { useInputFieldNamesByStatus } from 'features/nodes/hooks/useInputFieldNamesByStatus';
import { useOutputFieldNames } from 'features/nodes/hooks/useOutputFieldNames';
import { useWithFooter } from 'features/nodes/hooks/useWithFooter';
import { memo } from 'react';

import { InputFieldEditModeNodes } from './fields/InputFieldEditModeNodes';
import InvocationNodeFooter from './InvocationNodeFooter';
import InvocationNodeHeader from './InvocationNodeHeader';

type Props = {
  nodeId: string;
  isOpen: boolean;
  label: string;
  type: string;
  selected: boolean;
};

const InvocationNode = ({ nodeId, isOpen, label, type, selected }: Props) => {
  const fieldNames = useInputFieldNamesByStatus(nodeId);
  const withFooter = useWithFooter(nodeId);
  const outputFieldNames = useOutputFieldNames(nodeId);

  return (
    <NodeWrapper nodeId={nodeId} selected={selected}>
      <InvocationNodeHeader nodeId={nodeId} isOpen={isOpen} label={label} selected={selected} type={type} />
      {isOpen && (
        <>
          <Flex
            layerStyle="nodeBody"
            flexDirection="column"
            w="full"
            h="full"
            py={2}
            gap={1}
            borderBottomRadius={withFooter ? 0 : 'base'}
          >
            <Flex flexDir="column" px={2} w="full" h="full">
              <Grid gridTemplateColumns="1fr auto" gridAutoRows="1fr">
                {fieldNames.connectionFields.map((fieldName, i) => (
                  <GridItem gridColumnStart={1} gridRowStart={i + 1} key={`${nodeId}.${fieldName}.input-field`}>
                    <InputFieldGate nodeId={nodeId} fieldName={fieldName}>
                      <InputFieldEditModeNodes nodeId={nodeId} fieldName={fieldName} />
                    </InputFieldGate>
                  </GridItem>
                ))}
                {outputFieldNames.map((fieldName, i) => (
                  <GridItem gridColumnStart={2} gridRowStart={i + 1} key={`${nodeId}.${fieldName}.output-field`}>
                    <OutputFieldGate nodeId={nodeId} fieldName={fieldName}>
                      <OutputFieldNodesEditorView nodeId={nodeId} fieldName={fieldName} />
                    </OutputFieldGate>
                  </GridItem>
                ))}
              </Grid>
              {fieldNames.anyOrDirectFields.map((fieldName) => (
                <InputFieldGate key={`${nodeId}.${fieldName}.input-field`} nodeId={nodeId} fieldName={fieldName}>
                  <InputFieldEditModeNodes nodeId={nodeId} fieldName={fieldName} />
                </InputFieldGate>
              ))}
              {fieldNames.missingFields.map((fieldName) => (
                <InputFieldGate key={`${nodeId}.${fieldName}.input-field`} nodeId={nodeId} fieldName={fieldName}>
                  <InputFieldEditModeNodes nodeId={nodeId} fieldName={fieldName} />
                </InputFieldGate>
              ))}
            </Flex>
          </Flex>
          {withFooter && <InvocationNodeFooter nodeId={nodeId} />}
        </>
      )}
    </NodeWrapper>
  );
};

export default memo(InvocationNode);
