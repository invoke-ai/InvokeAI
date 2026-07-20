import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch } from 'app/store/storeHooks';
import type { SetNodeVideoFieldVideoDndTargetData } from 'features/dnd/dnd';
import { setNodeVideoFieldVideoDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { fieldVideoValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { VideoFieldInputInstance, VideoFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetVideoDTOQuery } from 'services/api/endpoints/videos';
import { $isConnected } from 'services/events/stores';

import type { FieldComponentProps } from './types';
import { isVideoMissingError } from './videoFieldErrors';

/**
 * Counterpart to ImageFieldInputComponent for VideoField inputs. Shows the video's WebP
 * thumbnail (the first decoded frame at the gallery-default size — same image the gallery
 * grid uses), with a small dimensions badge in the corner. Users drop a video from the
 * gallery onto the field; the drop is handled by setNodeVideoFieldVideoDndTarget.
 */
const VideoFieldInputComponent = (props: FieldComponentProps<VideoFieldInputInstance, VideoFieldInputTemplate>) => {
  const { t } = useTranslation();
  const { nodeId, field } = props;
  const dispatch = useAppDispatch();
  const isConnected = useStore($isConnected);

  const { currentData: videoDTO, error } = useGetVideoDTOQuery(field.value?.video_name ?? skipToken);

  const handleReset = useCallback(() => {
    dispatch(
      fieldVideoValueChanged({
        nodeId,
        fieldName: field.name,
        value: undefined,
      })
    );
  }, [dispatch, field.name, nodeId]);

  const dndTargetData = useMemo<SetNodeVideoFieldVideoDndTargetData>(
    () => setNodeVideoFieldVideoDndTarget.getData({ fieldIdentifier: { nodeId, fieldName: field.name } }),
    [field.name, nodeId]
  );

  // If the referenced video was deleted while disconnected, drop the stale reference once
  // we reconnect. Only a confirmed 404 means the video is gone — transient network, auth
  // (401/403), or server (5xx) failures must not silently clear the user's input.
  useEffect(() => {
    if (isConnected && isVideoMissingError(error)) {
      handleReset();
    }
  }, [handleReset, isConnected, error]);

  return (
    <Flex position="relative" className={NO_DRAG_CLASS} w="full" h={32} alignItems="stretch">
      {!videoDTO && (
        <Flex
          w="full"
          h="auto"
          alignItems="center"
          justifyContent="center"
          borderRadius="base"
          borderWidth={1}
          borderStyle="dashed"
        >
          <Text fontSize="sm" color="base.300">
            {t('gallery.drop')}
          </Text>
        </Flex>
      )}
      {videoDTO && (
        <>
          <Flex borderRadius="base" borderWidth={1} borderStyle="solid" overflow="hidden">
            <Image src={videoDTO.thumbnail_url} objectFit="contain" maxW="full" maxH="full" />
          </Flex>
          <Text
            position="absolute"
            background="base.900"
            color="base.50"
            fontSize="sm"
            fontWeight="semibold"
            insetInlineEnd={1}
            insetBlockEnd={1}
            opacity={0.7}
            px={2}
            borderRadius="base"
            pointerEvents="none"
          >{`${videoDTO.width}x${videoDTO.height}`}</Text>
        </>
      )}
      <DndDropTarget
        dndTarget={setNodeVideoFieldVideoDndTarget}
        dndTargetData={dndTargetData}
        label={t('gallery.drop')}
      />
    </Flex>
  );
};

export default memo(VideoFieldInputComponent);
