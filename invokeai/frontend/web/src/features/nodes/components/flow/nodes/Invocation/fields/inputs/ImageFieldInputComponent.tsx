import { Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch } from 'app/store/storeHooks';
import { UploadImageButton } from 'common/hooks/useImageUploadButton';
import type { SetNodeImageFieldImageDndTargetData } from 'features/dnd/dnd';
import { setNodeImageFieldImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { ImageFieldInputInstance, ImageFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

import type { FieldComponentProps } from './types';

const ImageFieldInputComponent = (props: FieldComponentProps<ImageFieldInputInstance, ImageFieldInputTemplate>) => {
  const { t } = useTranslation();
  const { nodeId, field, fieldTemplate } = props;
  const dispatch = useAppDispatch();
  const isConnected = useStore($isConnected);

  const { currentData: imageDTO, isError } = useGetImageDTOQuery(field.value?.image_name ?? skipToken);
  const handleReset = useCallback(() => {
    dispatch(
      fieldImageValueChanged({
        nodeId,
        fieldName: field.name,
        value: undefined,
      })
    );
  }, [dispatch, field.name, nodeId]);

  const dndTargetData = useMemo<SetNodeImageFieldImageDndTargetData>(
    () =>
      setNodeImageFieldImageDndTarget.getData(
        { fieldIdentifier: { nodeId, fieldName: field.name } },
        field.value?.image_name
      ),
    [field, nodeId]
  );

  useEffect(() => {
    if (isConnected && isError) {
      handleReset();
    }
  }, [handleReset, isConnected, isError]);

  const onUpload = useCallback(
    (imageDTO: ImageDTO) => {
      dispatch(
        fieldImageValueChanged({
          nodeId,
          fieldName: field.name,
          value: imageDTO,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <Flex position="relative" className={NO_DRAG_CLASS} w="full" h={32} alignItems="stretch">
      {!imageDTO && (
        <UploadImageButton
          w="full"
          h="auto"
          isError={fieldTemplate.required && !field.value}
          onUpload={onUpload}
          fontSize={24}
        />
      )}
      {imageDTO && (
        <>
          <Flex borderRadius="base" borderWidth={1} borderStyle="solid">
            <DndImage imageDTO={imageDTO} asThumbnail />
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
          >{`${imageDTO.width}x${imageDTO.height}`}</Text>
        </>
      )}
      <DndDropTarget
        dndTarget={setNodeImageFieldImageDndTarget}
        dndTargetData={dndTargetData}
        label={t('gallery.drop')}
      />
    </Flex>
  );
};

export default memo(ImageFieldInputComponent);

const UploadElement = memo(() => {
  const { t } = useTranslation();
  return (
    <Flex h={16} w="full" alignItems="center" justifyContent="center">
      <Text fontSize={16} fontWeight="semibold">
        {t('gallery.dropOrUpload')}
      </Text>
    </Flex>
  );
});

UploadElement.displayName = 'UploadElement';
