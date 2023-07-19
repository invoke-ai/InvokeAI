import { Flex, Image } from '@chakra-ui/react';
import { NodeProps } from 'react-flow-renderer';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from 'app/store/store';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import IAINodeHeader from './IAINode/IAINodeHeader';
import IAINodeResizer, { OnResize } from './IAINode/IAINodeResizer';
import NodeWrapper from './NodeWrapper';
import { nodeSizeChanged } from '../store/nodesSlice';

const ProgressImageNode = (props: NodeProps) => {
  const progressImage = useSelector(
    (state: RootState) => state.system.progressImage
  );
  const nodeSize = useSelector((state: RootState) => state.nodes.nodeSize);
  const dispatch = useDispatch();
  const { selected } = props;

  const handleResize: OnResize = (_, { width, height }) => {
    const newSize =
      width > height ? { width, height: width } : { width: height, height };
    dispatch(nodeSizeChanged(newSize));
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
              width: nodeSize.width,
              height: nodeSize.height,
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
