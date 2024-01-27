import { Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import NodeWrapper from 'features/nodes/components/flow/nodes/common/NodeWrapper';
import { useAnyOrDirectInputFieldNames } from 'features/nodes/hooks/useAnyOrDirectInputFieldNames';
import { useConnectionInputFieldNames } from 'features/nodes/hooks/useConnectionInputFieldNames';
import { useOutputFieldNames } from 'features/nodes/hooks/useOutputFieldNames';
import { useWithFooter } from 'features/nodes/hooks/useWithFooter';
import { memo } from 'react';

import InputField from './fields/InputField';
import OutputField from './fields/OutputField';
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
  const inputConnectionFieldNames = useConnectionInputFieldNames(nodeId);
  const inputAnyOrDirectFieldNames = useAnyOrDirectInputFieldNames(nodeId);
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
                {inputConnectionFieldNames.map((fieldName, i) => (
                  <GridItem gridColumnStart={1} gridRowStart={i + 1} key={`${nodeId}.${fieldName}.input-field`}>
                    <InputField nodeId={nodeId} fieldName={fieldName} />
                  </GridItem>
                ))}
                {outputFieldNames.map((fieldName, i) => (
                  <GridItem gridColumnStart={2} gridRowStart={i + 1} key={`${nodeId}.${fieldName}.output-field`}>
                    <OutputField nodeId={nodeId} fieldName={fieldName} />
                  </GridItem>
                ))}
              </Grid>
              {inputAnyOrDirectFieldNames.map((fieldName) => (
                <InputField key={`${nodeId}.${fieldName}.input-field`} nodeId={nodeId} fieldName={fieldName} />
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
