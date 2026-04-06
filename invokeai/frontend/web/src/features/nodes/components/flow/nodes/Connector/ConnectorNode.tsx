import { Box, Icon, Menu, MenuButton, MenuItem, MenuList, Portal, useGlobalMenuClose } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { Node, NodeProps } from '@xyflow/react';
import { Handle, Position } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import NonInvocationNodeWrapper from 'features/nodes/components/flow/nodes/common/NonInvocationNodeWrapper';
import { $templates, connectorDeleted } from 'features/nodes/store/nodesSlice';
import { selectEdges, selectNodes } from 'features/nodes/store/selectors';
import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  getConnectorDeletionSpliceConnections,
} from 'features/nodes/store/util/connectorTopology';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import type { ConnectorNodeData } from 'features/nodes/types/invocation';
import type { MouseEvent as ReactMouseEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { PiDotOutlineFill, PiTrashBold } from 'react-icons/pi';

const handleSx = {
  w: 3,
  h: 3,
  borderRadius: 'full',
  borderWidth: 2,
  borderColor: 'base.900',
  bg: 'base.100',
};

const ConnectorNode = ({ id, selected }: NodeProps<Node<ConnectorNodeData>>) => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector(selectNodes);
  const edges = useAppSelector(selectEdges);
  const templates = useStore($templates);
  const [menuState, setMenuState] = useState<{ x: number; y: number } | null>(null);

  const spliceConnections = useMemo(
    () => getConnectorDeletionSpliceConnections(id, nodes, edges, templates, validateConnection),
    [edges, id, nodes, templates]
  );

  useGlobalMenuClose(
    useCallback(() => {
      setMenuState(null);
    }, [])
  );

  const onContextMenu = useCallback((e: ReactMouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setMenuState({ x: e.pageX, y: e.pageY });
  }, []);

  const onCloseMenu = useCallback(() => {
    setMenuState(null);
  }, []);

  const onDelete = useCallback(() => {
    if (!spliceConnections) {
      return;
    }
    dispatch(connectorDeleted({ connectorId: id, spliceConnections }));
    setMenuState(null);
  }, [dispatch, id, spliceConnections]);

  return (
    <>
      <NonInvocationNodeWrapper nodeId={id} selected={selected} width={16}>
        <Box
          onContextMenu={onContextMenu}
          position="relative"
          w={16}
          h={16}
          display="flex"
          alignItems="center"
          justifyContent="center"
          layerStyle="nodeBody"
          borderRadius="full"
          borderWidth={selected ? 2 : 1}
          borderColor={selected ? 'base.200' : 'base.700'}
          bg="base.850"
        >
          <Handle type="target" id={CONNECTOR_INPUT_HANDLE} position={Position.Left} style={handleSx} />
          <Icon as={PiDotOutlineFill} boxSize={8} color="base.100" />
          <Handle type="source" id={CONNECTOR_OUTPUT_HANDLE} position={Position.Right} style={handleSx} />
        </Box>
      </NonInvocationNodeWrapper>
      <Portal>
        <Menu isOpen={menuState !== null} onClose={onCloseMenu} placement="auto-start" gutter={0}>
          <MenuButton
            aria-hidden
            position="absolute"
            left={menuState?.x ?? -9999}
            top={menuState?.y ?? -9999}
            w={1}
            h={1}
            pointerEvents="none"
            bg="transparent"
          />
          <MenuList>
            <MenuItem icon={<PiTrashBold />} onClick={onDelete} isDisabled={!spliceConnections} isDestructive>
              Delete Connector
            </MenuItem>
          </MenuList>
        </Menu>
      </Portal>
    </>
  );
};

export default memo(ConnectorNode);
