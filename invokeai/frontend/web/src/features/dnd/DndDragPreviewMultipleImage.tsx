import type { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { setCustomNativeDragPreview } from '@atlaskit/pragmatic-drag-and-drop/element/set-custom-native-drag-preview';
import { Flex, Heading } from '@invoke-ai/ui-library';
import type { MultipleImageDndSourceData } from 'features/dnd/dnd';
import { DND_IMAGE_DRAG_PREVIEW_SIZE, preserveOffsetOnSourceFallbackCentered } from 'features/dnd/util';
import { memo } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';
import type { Param0 } from 'tsafe';

const DndDragPreviewMultipleImage = memo(({ imageDTOs }: { imageDTOs: ImageDTO[] }) => {
  const { t } = useTranslation();
  return (
    <Flex
      w={DND_IMAGE_DRAG_PREVIEW_SIZE}
      h={DND_IMAGE_DRAG_PREVIEW_SIZE}
      alignItems="center"
      justifyContent="center"
      flexDir="column"
      bg="base.900"
      borderRadius="base"
    >
      <Heading>{imageDTOs.length}</Heading>
      <Heading size="sm">{t('parameters.images')}</Heading>
    </Flex>
  );
});

DndDragPreviewMultipleImage.displayName = 'DndDragPreviewMultipleImage';

export type DndDragPreviewMultipleImageState = {
  type: 'multiple-image';
  container: HTMLElement;
  imageDTOs: ImageDTO[];
};

export const createMultipleImageDragPreview = (arg: DndDragPreviewMultipleImageState) =>
  createPortal(<DndDragPreviewMultipleImage imageDTOs={arg.imageDTOs} />, arg.container);

type SetMultipleDragPreviewArg = {
  multipleImageDndData: MultipleImageDndSourceData;
  setDragPreviewState: (dragPreviewState: DndDragPreviewMultipleImageState | null) => void;
  onGenerateDragPreviewArgs: Param0<Param0<typeof draggable>['onGenerateDragPreview']>;
};

export const setMultipleImageDragPreview = ({
  multipleImageDndData,
  onGenerateDragPreviewArgs,
  setDragPreviewState,
}: SetMultipleDragPreviewArg) => {
  const { nativeSetDragImage, source, location } = onGenerateDragPreviewArgs;
  setCustomNativeDragPreview({
    render({ container }) {
      setDragPreviewState({ type: 'multiple-image', container, imageDTOs: multipleImageDndData.payload.imageDTOs });
      return () => setDragPreviewState(null);
    },
    nativeSetDragImage,
    getOffset: preserveOffsetOnSourceFallbackCentered({
      element: source.element,
      input: location.current.input,
    }),
  });
};
