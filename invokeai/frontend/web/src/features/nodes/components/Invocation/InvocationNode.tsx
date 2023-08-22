import { Flex } from '@chakra-ui/react';
import { useFieldNames, useWithFooter } from 'features/nodes/hooks/useNodeData';
import { memo } from 'react';
import InputField from '../fields/InputField';
import OutputField from '../fields/OutputField';
import NodeFooter from './NodeFooter';
import NodeHeader from './NodeHeader';
import NodeWrapper from './NodeWrapper';

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
      <NodeHeader
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
            className={'nopan'}
            sx={{
              cursor: 'auto',
              flexDirection: 'column',
              w: 'full',
              h: 'full',
              py: 1,
              gap: 1,
              borderBottomRadius: withFooter ? 0 : 'base',
            }}
          >
            <Flex
              className="nopan"
              sx={{ flexDir: 'column', px: 2, w: 'full', h: 'full' }}
            >
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
          {withFooter && <NodeFooter nodeId={nodeId} />}
        </>
      )}
    </NodeWrapper>
  );
};

export default memo(InvocationNode);
