import { Flex } from '@chakra-ui/react';
import {
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { map, some } from 'lodash-es';
import { memo, useMemo } from 'react';
import { NodeProps } from 'reactflow';
import InputField from '../fields/InputField';
import OutputField from '../fields/OutputField';
import NodeFooter, { FOOTER_FIELDS } from './NodeFooter';
import NodeHeader from './NodeHeader';
import NodeWrapper from './NodeWrapper';

type Props = {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
};

const InvocationNode = ({ nodeProps, nodeTemplate }: Props) => {
  const { id: nodeId, data } = nodeProps;
  const { inputs, outputs, isOpen } = data;

  const inputFields = useMemo(
    () => map(inputs).filter((i) => i.name !== 'is_intermediate'),
    [inputs]
  );
  const outputFields = useMemo(() => map(outputs), [outputs]);

  const withFooter = useMemo(
    () => some(outputs, (output) => FOOTER_FIELDS.includes(output.type)),
    [outputs]
  );

  return (
    <NodeWrapper nodeProps={nodeProps}>
      <NodeHeader nodeProps={nodeProps} nodeTemplate={nodeTemplate} />
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
              {outputFields.map((field) => (
                <OutputField
                  key={`${nodeId}.${field.id}.input-field`}
                  nodeProps={nodeProps}
                  nodeTemplate={nodeTemplate}
                  field={field}
                />
              ))}
              {inputFields.map((field) => (
                <InputField
                  key={`${nodeId}.${field.id}.input-field`}
                  nodeProps={nodeProps}
                  nodeTemplate={nodeTemplate}
                  field={field}
                />
              ))}
            </Flex>
          </Flex>
          {withFooter && (
            <NodeFooter nodeProps={nodeProps} nodeTemplate={nodeTemplate} />
          )}
        </>
      )}
    </NodeWrapper>
  );
};

export default memo(InvocationNode);
