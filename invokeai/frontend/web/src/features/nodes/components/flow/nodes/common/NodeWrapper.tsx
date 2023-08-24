import {
  Box,
  ChakraProps,
  useColorModeValue,
  useToken,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  DRAG_HANDLE_CLASSNAME,
  NODE_WIDTH,
} from 'features/nodes/types/constants';
import { contextMenusClosed } from 'features/ui/store/uiSlice';
import { PropsWithChildren, memo, useCallback } from 'react';

type NodeWrapperProps = PropsWithChildren & {
  nodeId: string;
  selected: boolean;
  width?: NonNullable<ChakraProps['sx']>['w'];
};

const NodeWrapper = (props: NodeWrapperProps) => {
  const { width, children, selected } = props;

  const [
    nodeSelectedOutlineLight,
    nodeSelectedOutlineDark,
    shadowsXl,
    shadowsBase,
  ] = useToken('shadows', [
    'nodeSelectedOutline.light',
    'nodeSelectedOutline.dark',
    'shadows.xl',
    'shadows.base',
  ]);

  const dispatch = useAppDispatch();

  const shadow = useColorModeValue(
    nodeSelectedOutlineLight,
    nodeSelectedOutlineDark
  );

  const opacity = useAppSelector((state) => state.nodes.nodeOpacity);

  const handleClick = useCallback(() => {
    dispatch(contextMenusClosed());
  }, [dispatch]);

  return (
    <Box
      onClick={handleClick}
      className={DRAG_HANDLE_CLASSNAME}
      sx={{
        h: 'full',
        position: 'relative',
        borderRadius: 'base',
        w: width ?? NODE_WIDTH,
        transitionProperty: 'common',
        transitionDuration: '0.1s',
        shadow: selected ? shadow : undefined,
        cursor: 'grab',
        opacity,
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          insetInlineEnd: 0,
          bottom: 0,
          insetInlineStart: 0,
          borderRadius: 'base',
          pointerEvents: 'none',
          shadow: `${shadowsXl}, ${shadowsBase}, ${shadowsBase}`,
          zIndex: -1,
        }}
      />
      {children}
    </Box>
  );
};

export default memo(NodeWrapper);
