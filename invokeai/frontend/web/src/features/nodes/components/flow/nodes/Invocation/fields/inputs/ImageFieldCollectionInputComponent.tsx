import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Grid, GridItem, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { UploadMultipleImageButton } from 'common/hooks/useImageUploadButton';
import type { AddImagesToNodeImageFieldCollection } from 'features/dnd/dnd';
import { addImagesToNodeImageFieldCollectionDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImageFromImageName } from 'features/dnd/DndImageFromImageName';
import { useFieldIsInvalid } from 'features/nodes/hooks/useFieldIsInvalid';
import { fieldImageCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import type { ImageFieldCollectionInputInstance, ImageFieldCollectionInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import type { FieldComponentProps } from './types';

const sx = {
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
    borderWidth: 1,
  },
} satisfies SystemStyleObject;

export const ImageFieldCollectionInputComponent = memo(
  (props: FieldComponentProps<ImageFieldCollectionInputInstance, ImageFieldCollectionInputTemplate>) => {
    const { t } = useTranslation();
    const { nodeId, field } = props;
    const dispatch = useAppDispatch();
    const isInvalid = useFieldIsInvalid(nodeId, field.name);

    const onReset = useCallback(() => {
      dispatch(
        fieldImageCollectionValueChanged({
          nodeId,
          fieldName: field.name,
          value: [],
        })
      );
    }, [dispatch, field.name, nodeId]);

    const dndTargetData = useMemo<AddImagesToNodeImageFieldCollection>(
      () => addImagesToNodeImageFieldCollectionDndTarget.getData({ fieldIdentifer: { nodeId, fieldName: field.name } }),
      [field, nodeId]
    );

    const onUpload = useCallback(
      (imageDTOs: ImageDTO[]) => {
        dispatch(
          fieldImageCollectionValueChanged({
            nodeId,
            fieldName: field.name,
            value: imageDTOs,
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
        {(!field.value || field.value.length === 0) && (
          <UploadMultipleImageButton
            w="full"
            h="auto"
            isError={isInvalid}
            onUpload={onUpload}
            fontSize={24}
            variant="outline"
          />
        )}
        {field.value && field.value.length > 0 && (
          <>
            <Grid
              className="nopan"
              borderRadius="base"
              w="full"
              h="full"
              templateColumns={`repeat(${Math.min(field.value.length, 3)}, 1fr)`}
              gap={1}
              sx={sx}
              data-error={isInvalid}
              p={1}
            >
              {field.value.map(({ image_name }) => (
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
            />
          </>
        )}
        <DndDropTarget
          dndTarget={addImagesToNodeImageFieldCollectionDndTarget}
          dndTargetData={dndTargetData}
          label={t('gallery.drop')}
        />
      </Flex>
    );
  }
);

ImageFieldCollectionInputComponent.displayName = 'ImageFieldCollectionInputComponent';
