import { Flex, Icon } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { makeTemplateSelector } from 'features/nodes/store/util/makeTemplateSelector';
import { InvocationNodeData } from 'features/nodes/types/types';
import { map } from 'lodash-es';
import { memo, useMemo } from 'react';
import { FaExclamationCircle } from 'react-icons/fa';
import { NodeProps } from 'reactflow';
import NodeCollapseButton from '../Invocation/NodeCollapseButton';
import NodeCollapsedHandles from '../Invocation/NodeCollapsedHandles';
import NodeFooter from '../Invocation/NodeFooter';
import NodeNotesEdit from '../Invocation/NodeNotesEdit';
import NodeStatusIndicator from '../Invocation/NodeStatusIndicator';
import NodeTitle from '../Invocation/NodeTitle';
import NodeWrapper from '../Invocation/NodeWrapper';
import InputField from '../fields/InputField';
import OutputField from '../fields/OutputField';

const InvocationNode = (props: NodeProps<InvocationNodeData>) => {
  const { id: nodeId, data } = props;
  const { type, inputs, outputs, isOpen } = data;

  const templateSelector = useMemo(() => makeTemplateSelector(type), [type]);
  const nodeTemplate = useAppSelector(templateSelector);
  const inputFields = useMemo(
    () => map(inputs).filter((i) => i.name !== 'is_intermediate'),
    [inputs]
  );
  const outputFields = useMemo(() => map(outputs), [outputs]);

  if (!nodeTemplate) {
    return (
      <NodeWrapper nodeProps={props}>
        <Flex
          layerStyle="nodeBody"
          className="nopan"
          sx={{
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'auto',
            w: 'full',
            h: 'full',
          }}
        >
          <Icon
            as={FaExclamationCircle}
            sx={{
              boxSize: 32,
              color: 'base.600',
              _dark: { color: 'base.400' },
            }}
          ></Icon>
        </Flex>
      </NodeWrapper>
    );
  }

  return (
    <NodeWrapper nodeProps={props}>
      <Flex
        layerStyle="nodeHeader"
        sx={{
          borderTopRadius: 'base',
          borderBottomRadius: isOpen ? 0 : 'base',
          alignItems: 'center',
          justifyContent: 'space-between',
          h: 8,
          textAlign: 'center',
          fontWeight: 600,
          color: 'base.700',
          _dark: { color: 'base.200' },
        }}
      >
        <NodeCollapseButton nodeProps={props} />
        <NodeTitle nodeData={props.data} title={nodeTemplate.title} />
        <Flex alignItems="center">
          <NodeStatusIndicator nodeProps={props} />
          <NodeNotesEdit nodeProps={props} nodeTemplate={nodeTemplate} />
        </Flex>
        {!isOpen && (
          <NodeCollapsedHandles nodeProps={props} nodeTemplate={nodeTemplate} />
        )}
      </Flex>
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
            }}
          >
            <Flex
              className="nopan"
              sx={{ flexDir: 'column', px: 2, w: 'full', h: 'full' }}
            >
              {outputFields.map((field) => (
                <OutputField
                  key={`${nodeId}.${field.id}.input-field`}
                  nodeProps={props}
                  nodeTemplate={nodeTemplate}
                  field={field}
                />
              ))}
              {inputFields.map((field) => (
                <InputField
                  key={`${nodeId}.${field.id}.input-field`}
                  nodeProps={props}
                  nodeTemplate={nodeTemplate}
                  field={field}
                />
              ))}
            </Flex>
          </Flex>
          <NodeFooter nodeProps={props} nodeTemplate={nodeTemplate} />
        </>
      )}
    </NodeWrapper>
  );
};

export default memo(InvocationNode);
