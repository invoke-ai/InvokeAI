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
import { removeImageFromNodeImageFieldCollectionAction } from 'features/imageActions/actions';
import { useFieldIsInvalid } from 'features/nodes/hooks/useFieldIsInvalid';
import { fieldImageCollectionValueChanged } from 'features/nodes/store/nodesSlice';
import type { ImageFieldCollectionInputInstance, ImageFieldCollectionInputTemplate } from 'features/nodes/types/field';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiExclamationMarkBold } from 'react-icons/pi';
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

    const isInvalid = useFieldIsInvalid(nodeId, field.name);

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
      (imageName: string) => {
        removeImageFromNodeImageFieldCollectionAction({
          imageName,
          fieldIdentifier: { nodeId, fieldName: field.name },
          dispatch: store.dispatch,
          getState: store.getState,
        });
      },
      [field.name, nodeId, store.dispatch, store.getState]
    );

    return (
      <Flex
        className="nodrag"
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
            variant="outline"
          />
        )}
        {field.value && field.value.length > 0 && (
          <Box w="full" h="auto" p={1} sx={sx} data-error={isInvalid} borderRadius="base">
            <OverlayScrollbarsComponent
              className="nowheel"
              defer
              style={overlayScrollbarsStyles}
              options={overlayscrollbarsOptions}
            >
              <Grid w="full" h="full" templateColumns="repeat(4, 1fr)" gap={1}>
                {field.value.map(({ image_name }) => (
                  <GridItem key={image_name} position="relative" className="nodrag">
                    <ImageGridItemContent imageName={image_name} onRemoveImage={onRemoveImage} />
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
  ({ imageName, onRemoveImage }: { imageName: string; onRemoveImage: (imageName: string) => void }) => {
    const query = useGetImageDTOQuery(imageName);
    const onClickRemove = useCallback(() => {
      onRemoveImage(imageName);
    }, [imageName, onRemoveImage]);

    if (query.isLoading) {
      return <IAINoContentFallbackWithSpinner />;
    }

    if (!query.data) {
      return <IAINoContentFallback icon={<PiExclamationMarkBold />} />;
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
          icon={<PiArrowCounterClockwiseBold />}
          tooltip="Reset Image"
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
