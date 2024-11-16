import { Flex, Grid, GridItem, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { UploadMultipleImageButton } from 'common/hooks/useImageUploadButton';
import type { AddImagesToNodeImageFieldCollection } from 'features/dnd/dnd';
import { addImagesToNodeImageFieldCollectionDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImageFromImageName } from 'features/dnd/DndImageFromImageName';
import { fieldImageCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import type { ImageFieldCollectionInputInstance, ImageFieldCollectionInputTemplate } from 'features/nodes/types/field';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import type { FieldComponentProps } from './types';

export const ImageFieldCollectionInputComponent = memo(
  (props: FieldComponentProps<ImageFieldCollectionInputInstance, ImageFieldCollectionInputTemplate>) => {
    const { t } = useTranslation();
    const { nodeId, field, fieldTemplate } = props;
    const dispatch = useAppDispatch();
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

    const isInvalid = useMemo(() => {
      if (!field.value) {
        if (fieldTemplate.required) {
          return true;
        }
      } else if (fieldTemplate.minLength !== undefined && field.value.length < fieldTemplate.minLength) {
        return true;
      } else if (fieldTemplate.maxLength !== undefined && field.value.length > fieldTemplate.maxLength) {
        return true;
      }
      return false;
    }, [field.value, fieldTemplate.maxLength, fieldTemplate.minLength, fieldTemplate.required]);

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
            <Grid className="nopan" w="full" h="full" templateColumns="repeat(3, 1fr)" gap={2}>
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
