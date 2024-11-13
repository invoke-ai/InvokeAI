import { Box, Flex, Grid, GridItem, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type { AddImagesToBatchImageInputNodeDndTargetData } from 'features/dnd/dnd';
import { addImagesToBatchImageInputNodeDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImageFromImageName } from 'features/dnd/DndImageFromImageName';
import NodeCollapseButton from 'features/nodes/components/flow/nodes/common/NodeCollapseButton';
import NodeTitle from 'features/nodes/components/flow/nodes/common/NodeTitle';
import NodeWrapper from 'features/nodes/components/flow/nodes/common/NodeWrapper';
import FieldHandle from 'features/nodes/components/flow/nodes/Invocation/fields/FieldHandle';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { batchImageInputNodeReset } from 'features/nodes/store/nodesSlice';
import { imageBatchOutputFieldTemplate } from 'features/nodes/types/field';
import type { ImageBatchNodeData } from 'features/nodes/types/invocation';
import { memo, useCallback, useMemo } from 'react';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import type { NodeProps } from 'reactflow';

export const BatchImageInputNode = memo((props: NodeProps<ImageBatchNodeData>) => {
  const { id: nodeId, data, selected } = props;
  const { images, isOpen } = data;
  const dispatch = useAppDispatch();
  const onReset = useCallback(() => {
    dispatch(batchImageInputNodeReset({ nodeId }));
  }, [dispatch, nodeId]);
  const targetData = useMemo<AddImagesToBatchImageInputNodeDndTargetData>(
    () => addImagesToBatchImageInputNodeDndTarget.getData({ nodeId }),
    [nodeId]
  );

  return (
    <NodeWrapper nodeId={nodeId} selected={selected}>
      <Flex
        layerStyle="nodeHeader"
        borderTopRadius="base"
        borderBottomRadius={isOpen ? 0 : 'base'}
        alignItems="center"
        justifyContent="space-between"
        h={8}
      >
        <NodeCollapseButton nodeId={nodeId} isOpen={isOpen} />
        <NodeTitle nodeId={nodeId} title="Batch Image Input" />
        <Box minW={8} />
      </Flex>
      {isOpen && (
        <>
          <Flex
            position="relative"
            layerStyle="nodeBody"
            className="nopan"
            cursor="auto"
            flexDirection="column"
            borderBottomRadius="base"
            w="full"
            h="full"
            p={2}
            gap={1}
            minH={16}
          >
            <Grid className="nopan" w="full" h="full" templateColumns="repeat(3, 1fr)" gap={2}>
              {images.map(({ image_name }) => (
                <GridItem key={image_name}>
                  <DndImageFromImageName imageName={image_name} asThumbnail />
                </GridItem>
              ))}
            </Grid>
            <IconButton
              aria-label="reset"
              icon={<PiArrowCounterClockwiseBold />}
              position="absolute"
              top={0}
              insetInlineEnd={0}
              onClick={onReset}
              variant="ghost"
            />
          </Flex>
        </>
      )}
      <ImageBatchOutputField nodeId={nodeId} />

      <DndDropTarget
        dndTarget={addImagesToBatchImageInputNodeDndTarget}
        dndTargetData={targetData}
        label="Add to Batch"
      />
    </NodeWrapper>
  );
});

BatchImageInputNode.displayName = 'BatchImageInputNode';

const ImageBatchOutputField = memo(({ nodeId }: { nodeId: string }) => {
  const { isConnected, isConnectionInProgress, isConnectionStartField, validationResult, shouldDim } =
    useConnectionState({ nodeId, fieldName: 'images', kind: 'outputs' });

  return (
    <Flex
      position="absolute"
      minH={8}
      top="50%"
      translateY="-50%"
      insetInlineEnd={2}
      alignItems="center"
      opacity={shouldDim ? 0.5 : 1}
      transitionProperty="opacity"
      transitionDuration="0.1s"
      justifyContent="flex-end"
    >
      <FieldHandle
        fieldTemplate={imageBatchOutputFieldTemplate}
        handleType="source"
        isConnectionInProgress={isConnectionInProgress}
        isConnectionStartField={isConnectionStartField}
        validationResult={validationResult}
      />
    </Flex>
  );
});
ImageBatchOutputField.displayName = 'ImageBatchOutputField';
