import { Box, Flex, Text } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { makeTemplateSelector } from 'features/nodes/store/util/makeTemplateSelector';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { InvocationNodeData } from 'features/nodes/types/types';
import { map } from 'lodash-es';
import { memo, useMemo } from 'react';
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
          className={DRAG_HANDLE_CLASSNAME}
          layerStyle="nodeHeader"
          sx={{
            borderTopRadius: 'base',
            borderBottomRadius: isOpen ? 0 : 'base',
            alignItems: 'center',
            h: 8,
            fontWeight: 600,
            fontSize: 'sm',
          }}
        >
          <NodeCollapseButton nodeProps={props} />
          <Text
            sx={{
              w: 'full',
              textAlign: 'center',
              pe: 8,
              color: 'error.500',
              _dark: { color: 'error.300' },
            }}
          >
            {data.label ? `${data.label} (${data.type})` : data.type}
          </Text>
        </Flex>
        {isOpen && (
          <Flex
            layerStyle="nodeBody"
            sx={{
              userSelect: 'auto',
              flexDirection: 'column',
              w: 'full',
              h: 'full',
              p: 4,
              gap: 1,
              borderBottomRadius: 'base',
              fontSize: 'sm',
            }}
          >
            <Box>
              <Text as="span">Unknown node type: </Text>
              <Text as="span" fontWeight={600}>
                {data.type}
              </Text>
            </Box>
          </Flex>
        )}
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
