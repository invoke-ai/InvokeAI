import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import InvocationNodeFooter from './InvocationNodeFooter';
import InvocationNodeHeader from './InvocationNodeHeader';
import NodeWrapper from '../common/NodeWrapper';
import OutputField from './fields/OutputField';
import InputField from './fields/InputField';
import { useFieldNames } from 'features/nodes/hooks/useFieldNames';
import { useWithFooter } from 'features/nodes/hooks/useWithFooter';

type Props = {
  nodeId: string;
  isOpen: boolean;
  label: string;
  type: string;
  selected: boolean;
};

const InvocationNode = ({ nodeId, isOpen, label, type, selected }: Props) => {
  const inputFieldNames = useFieldNames(nodeId, 'input');
  const outputFieldNames = useFieldNames(nodeId, 'output');
  const withFooter = useWithFooter(nodeId);

  return (
    <NodeWrapper nodeId={nodeId} selected={selected}>
      <InvocationNodeHeader
        nodeId={nodeId}
        isOpen={isOpen}
        label={label}
        selected={selected}
        type={type}
      />
      {isOpen && (
        <>
          <Flex
            layerStyle="nodeBody"
            sx={{
              flexDirection: 'column',
              w: 'full',
              h: 'full',
              py: 1,
              gap: 1,
              borderBottomRadius: withFooter ? 0 : 'base',
            }}
          >
            <Flex sx={{ flexDir: 'column', px: 2, w: 'full', h: 'full' }}>
              {outputFieldNames.map((fieldName) => (
                <OutputField
                  key={`${nodeId}.${fieldName}.output-field`}
                  nodeId={nodeId}
                  fieldName={fieldName}
                />
              ))}
              {inputFieldNames.map((fieldName) => (
                <InputField
                  key={`${nodeId}.${fieldName}.input-field`}
                  nodeId={nodeId}
                  fieldName={fieldName}
                />
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
