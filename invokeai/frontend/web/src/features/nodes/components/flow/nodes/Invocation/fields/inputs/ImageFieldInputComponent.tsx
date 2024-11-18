import { Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch } from 'app/store/storeHooks';
import { UploadImageButton } from 'common/hooks/useImageUploadButton';
import type { SetNodeImageFieldImageDndTargetData } from 'features/dnd/dnd';
import { setNodeImageFieldImageDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import type { ImageFieldInputInstance, ImageFieldInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
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
    <Flex
      position="relative"
      className="nodrag"
      w="full"
      h="full"
      minH={16}
      alignItems="stretch"
      justifyContent="center"
    >
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
          <DndImage imageDTO={imageDTO} minW={8} minH={8} />
          <DndImageIcon
            onClick={handleReset}
            icon={imageDTO ? <PiArrowCounterClockwiseBold /> : undefined}
            tooltip="Reset Image"
            position="absolute"
            flexDir="column"
            top={1}
            insetInlineEnd={1}
            gap={1}
          />
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
