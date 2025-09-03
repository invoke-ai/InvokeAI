import { Editable, EditableInput, EditablePreview, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setFocusedRegion } from 'common/hooks/focus';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { canvasInstanceRemoved } from 'features/controlLayers/store/canvasesSlice';
import { selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectCanvasCount } from 'features/controlLayers/store/selectors';
import { useCurrentQueueItemDestination } from 'features/queue/hooks/useCurrentQueueItemDestination';
import ProgressBar from 'features/system/components/ProgressBar';
import type { MouseEvent } from 'react';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiX } from 'react-icons/pi';
import { useIsGenerationInProgress } from 'services/api/endpoints/queue';

import type { DockviewPanelParameters } from './auto-layout-context';

export const DockviewTabCanvasWorkspace = memo((props: IDockviewPanelHeaderProps<DockviewPanelParameters>) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isGenerationInProgress = useIsGenerationInProgress();
  const canvasSessionId = useAppSelector(selectCanvasSessionId);
  const canvasCount = useAppSelector(selectCanvasCount);
  const currentQueueItemDestination = useCurrentQueueItemDestination();

  const ref = useRef<HTMLDivElement>(null);
  const setActive = useCallback(() => {
    if (!props.api.isActive) {
      props.api.setActive();
    }
  }, [props.api]);

  useCallbackOnDragEnter(setActive, ref, 300);

  const onPointerDown = useCallback(() => {
    setFocusedRegion(props.params.focusRegion);
  }, [props.params.focusRegion]);

  const [title, setTitle] = useState(props.api.title || t(props.params.i18nKey));

  const handleClose = useCallback(
    (e: MouseEvent) => {
      e.stopPropagation();

      if (props.params.canvasId) {
        // Remove from Redux - this should also trigger panel removal
        dispatch(canvasInstanceRemoved({ canvasId: props.params.canvasId }));

        // Close the dockview panel
        props.api.close();
      }
    },
    [dispatch, props.params.canvasId, props.api]
  );

  const handleTitleChange = useCallback(
    (newTitle: string) => {
      const trimmedTitle = newTitle.trim();
      if (trimmedTitle.length > 0) {
        setTitle(trimmedTitle);
        // Update the dockview panel title
        props.api.setTitle(trimmedTitle);
      }
    },
    [props.api]
  );

  const handleTitleSubmit = useCallback(
    (newTitle: string) => {
      handleTitleChange(newTitle);
    },
    [handleTitleChange]
  );

  const handleTitleCancel = useCallback(() => {
    setTitle(props.api.title || t(props.params.i18nKey));
  }, [props.api.title, props.params.i18nKey, t]);

  // Show close button only if:
  // 1. This is a canvas panel with a canvasId (not launchpad or viewer)
  // 2. There's more than one canvas (can't close the last one)
  const canShowCloseButton = props.params.canvasId && canvasCount > 1;

  // Only allow renaming for canvas panels with canvasId (not launchpad or viewer)
  const canRename = Boolean(props.params.canvasId);

  return (
    <Flex ref={ref} position="relative" alignItems="center" h="full" onPointerDown={onPointerDown}>
      {canRename ? (
        <Editable
          value={title}
          onChange={setTitle}
          onSubmit={handleTitleSubmit}
          onCancel={handleTitleCancel}
          px={4}
          flex="1"
          fontSize="sm"
        >
          <EditablePreview cursor="pointer" borderRadius="md" userSelect="none" />
          <EditableInput fontSize="sm" />
        </Editable>
      ) : (
        <Text userSelect="none" px={4} flex="1" fontSize="sm">
          {title}
        </Text>
      )}
      {canShowCloseButton && (
        <IconButton aria-label="Close Canvas" icon={<PiX />} size="xs" variant="ghost" onClick={handleClose} mr={1} />
      )}
      {currentQueueItemDestination === canvasSessionId && isGenerationInProgress && (
        <ProgressBar position="absolute" bottom={0} left={0} right={0} h={1} borderRadius="none" />
      )}
    </Flex>
  );
});
DockviewTabCanvasWorkspace.displayName = 'DockviewTabCanvasWorkspace';
