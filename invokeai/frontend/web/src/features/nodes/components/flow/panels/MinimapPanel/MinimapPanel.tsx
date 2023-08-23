import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { Flex, useColorModeValue } from '@chakra-ui/react';
import { memo } from 'react';
import { MiniMap } from 'reactflow';

const MinimapPanel = () => {
  const miniMapStyle = useColorModeValue(
    {
      background: 'var(--invokeai-colors-base-200)',
    },
    {
      background: 'var(--invokeai-colors-base-500)',
    }
  );

  const shouldShowMinimapPanel = useAppSelector(
    (state: RootState) => state.nodes.shouldShowMinimapPanel
  );

  const nodeColor = useColorModeValue(
    'var(--invokeai-colors-accent-300)',
    'var(--invokeai-colors-accent-600)'
  );

  const maskColor = useColorModeValue(
    'var(--invokeai-colors-blackAlpha-300)',
    'var(--invokeai-colors-blackAlpha-600)'
  );

  return (
    <Flex sx={{ gap: 2, position: 'absolute', bottom: 2, insetInlineEnd: 2 }}>
      {shouldShowMinimapPanel && (
        <MiniMap
          pannable
          zoomable
          nodeBorderRadius={15}
          style={miniMapStyle}
          nodeColor={nodeColor}
          maskColor={maskColor}
        />
      )}
    </Flex>
  );
};

export default memo(MinimapPanel);
