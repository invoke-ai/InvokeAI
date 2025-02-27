import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import { getOverlayScrollbarsParams, overlayScrollbarsStyles } from 'common/components/OverlayScrollbars/constants';
import { UploadMultipleImageButton } from 'common/hooks/useImageUploadButton';
import { TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import type { AddImagesToNodeImageFieldCollection } from 'features/dnd/dnd';
import { addImagesToNodeImageFieldCollectionDndTarget } from 'features/dnd/dnd';
import { DndDropTarget } from 'features/dnd/DndDropTarget';
import { DndImage } from 'features/dnd/DndImage';
import { DndImageIcon } from 'features/dnd/DndImageIcon';
import { useInputFieldIsInvalid } from 'features/nodes/hooks/useInputFieldIsInvalid';
import { fieldImageCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import type { ImageField } from 'features/nodes/types/common';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import type { ImageFieldCollectionInputInstance, ImageFieldCollectionInputTemplate } from 'features/nodes/types/field';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiExclamationMarkBold, PiXBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

import type { FieldComponentProps } from './types';

const overlayscrollbarsOptions = getOverlayScrollbarsParams().options;

const sx = {
  borderWidth: 1,
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
  },
} satisfies SystemStyleObject;

export const ImageFieldCollectionInputComponent = memo(
  (props: FieldComponentProps<ImageFieldCollectionInputInstance, ImageFieldCollectionInputTemplate>) => {
    const { t } = useTranslation();
    const { nodeId, field } = props;
    const store = useAppStore();

    const isInvalid = useInputFieldIsInvalid(nodeId, field.name);

    const dndTargetData = useMemo<AddImagesToNodeImageFieldCollection>(
      () =>
        addImagesToNodeImageFieldCollectionDndTarget.getData({ fieldIdentifier: { nodeId, fieldName: field.name } }),
      [field, nodeId]
    );

    const onUpload = useCallback(
      (imageDTOs: ImageDTO[]) => {
        store.dispatch(
          fieldImageCollectionValueChanged({
            nodeId,
            fieldName: field.name,
            value: imageDTOs,
          })
        );
      },
      [store, nodeId, field.name]
    );

    const onRemoveImage = useCallback(
      (index: number) => {
        const newValue = field.value ? [...field.value] : [];
        newValue.splice(index, 1);
        store.dispatch(fieldImageCollectionValueChanged({ nodeId, fieldName: field.name, value: newValue }));
      },
      [field.name, field.value, nodeId, store]
    );

    return (
      <Flex
        className={NO_DRAG_CLASS}
        position="relative"
        w="full"
        h="full"
        minH={16}
        maxH={64}
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
            variant="ghost"
          />
        )}
        {field.value && field.value.length > 0 && (
          <Box w="full" h="auto" p={1} sx={sx} data-error={isInvalid} borderRadius="base">
            <OverlayScrollbarsComponent
              className={NO_WHEEL_CLASS}
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Grid w="full" h="full" templateColumns="repeat(4, 1fr)" gap={1}>
                {field.value.map((value, index) => (
                  <GridItem key={index} position="relative" className={NO_DRAG_CLASS}>
                    <ImageGridItemContent value={value} index={index} onRemoveImage={onRemoveImage} />
                  </GridItem>
                ))}
              </Grid>
            </OverlayScrollbarsComponent>
          </Box>
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

const ImageGridItemContent = memo(
  ({ value, index, onRemoveImage }: { value: ImageField; index: number; onRemoveImage: (index: number) => void }) => {
    const query = useGetImageDTOQuery(value.image_name);
    const onClickRemove = useCallback(() => {
      onRemoveImage(index);
    }, [index, onRemoveImage]);

    if (query.isLoading) {
      return <IAINoContentFallbackWithSpinner />;
    }

    if (!query.data) {
      return (
        <>
          <IAINoContentFallback icon={PiExclamationMarkBold} />
          <DndImageIcon
            onClick={onClickRemove}
            icon={<PiXBold />}
            tooltip="Remove Image from Collection"
            position="absolute"
            flexDir="column"
            top={1}
            insetInlineEnd={1}
            gap={1}
          />
        </>
      );
    }

    return (
      <>
        <DndImage
          imageDTO={query.data}
          asThumbnail
          objectFit="contain"
          w="full"
          h="full"
          aspectRatio="1/1"
          backgroundSize={8}
          backgroundImage={TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL}
        />
        <DndImageIcon
          onClick={onClickRemove}
          icon={<PiXBold />}
          tooltip="Remove Image from Collection"
          position="absolute"
          flexDir="column"
          top={1}
          insetInlineEnd={1}
          gap={1}
        />
      </>
    );
  }
);
ImageGridItemContent.displayName = 'ImageGridItemContent';
