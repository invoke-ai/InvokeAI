import { Flex, Image } from '@chakra-ui/react';
import { NodeProps } from 'reactflow';
import { InvocationValue } from '../types/types';

import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo, useEffect, useState } from 'react';
import IAINodeHeader from './IAINode/IAINodeHeader';
import IAINodeResizer, { OnResize } from './IAINode/IAINodeResizer';
import NodeWrapper from './NodeWrapper';

const ProgressImageNode = (props: NodeProps<InvocationValue>) => {
  const progressImage = useAppSelector((state) => state.system.progressImage);
  const { selected } = props;

  const [nodeSize, setNodeSize] = useState(() => {
    const storedSize = localStorage.getItem('nodeSize');
    return storedSize ? JSON.parse(storedSize) : { width: 512, height: 512 };
  });

  useEffect(() => {
    localStorage.setItem('nodeSize', JSON.stringify(nodeSize));
  }, [nodeSize]);

  const handleResize: OnResize = (_, { width, height }) => {
    setNodeSize({ width, height });
  };

  return (
    <NodeWrapper selected={selected}>
      <IAINodeHeader
        title="Progress Image"
        description="Displays the progress image in the Node Editor"
      />

      <Flex
        sx={{
          flexDirection: 'column',
          borderBottomRadius: 'md',
          p: 2,
          bg: 'base.200',
          _dark: { bg: 'base.800' },
          width: nodeSize.width,
          height: nodeSize.height,
          overflow: 'hidden',
          flexShrink: 0,
        }}
      >
        {progressImage ? (
          <Image
            src={progressImage.dataURL}
            sx={{
              w: 'full',
              h: 'full',
              objectFit: 'contain',
              maxWidth: '100%',
              maxHeight: '100%',
            }}
          />
        ) : (
          <Flex
            sx={{
              w: 'full',
              h: 'full',
              minW: 32,
              minH: 32,
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <IAINoContentFallback />
          </Flex>
        )}
      </Flex>
      <IAINodeResizer onResize={handleResize} />
    </NodeWrapper>
  );
};

export default memo(ProgressImageNode);
