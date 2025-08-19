import type { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { setCustomNativeDragPreview } from '@atlaskit/pragmatic-drag-and-drop/element/set-custom-native-drag-preview';
import { Flex, Heading } from '@invoke-ai/ui-library';
import type { MultipleVideoDndSourceData } from 'features/dnd/dnd';
import { DND_IMAGE_DRAG_PREVIEW_SIZE, preserveOffsetOnSourceFallbackCentered } from 'features/dnd/util';
import { memo } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import type { Param0 } from 'tsafe';

const DndDragPreviewMultipleVideo = memo(({ ids}: { ids: string[] }) => {
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
      <Heading>{ids.length}</Heading>
      <Heading size="sm">{t('parameters.videos')}</Heading>
    </Flex>
  );
});

DndDragPreviewMultipleVideo.displayName = 'DndDragPreviewMultipleVideo';

export type DndDragPreviewMultipleVideoState = {
  type: 'multiple-video';
  container: HTMLElement;
  ids: string[];
};

export const createMultipleVideoDragPreview = (arg: DndDragPreviewMultipleVideoState) =>
  createPortal(<DndDragPreviewMultipleVideo ids={arg.ids} />, arg.container);

type SetMultipleDragPreviewArg = {
  multipleVideoDndData: MultipleVideoDndSourceData;
  setDragPreviewState: (dragPreviewState: DndDragPreviewMultipleVideoState | null) => void;
  onGenerateDragPreviewArgs: Param0<Param0<typeof draggable>['onGenerateDragPreview']>;
};

export const setMultipleVideoDragPreview = ({
  multipleVideoDndData,
  onGenerateDragPreviewArgs,
  setDragPreviewState,
}: SetMultipleDragPreviewArg) => {
  const { nativeSetDragImage, source, location } = onGenerateDragPreviewArgs;
  setCustomNativeDragPreview({
    render({ container }) {
      setDragPreviewState({ type: 'multiple-video', container, ids: multipleVideoDndData.payload.ids});
      return () => setDragPreviewState(null);
    },
    nativeSetDragImage,
    getOffset: preserveOffsetOnSourceFallbackCentered({
      element: source.element,
      input: location.current.input,
    }),
  });
};
