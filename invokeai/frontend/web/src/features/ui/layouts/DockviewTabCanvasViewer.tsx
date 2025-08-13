import { Flex, Text } from '@invoke-ai/ui-library';
import { setFocusedRegion } from 'common/hooks/focus';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { useCurrentQueueItemDestination } from 'features/queue/hooks/useCurrentQueueItemDestination';
import ProgressBar from 'features/system/components/ProgressBar';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsGenerationInProgress } from 'services/api/endpoints/queue';

import type { DockviewPanelParameters } from './auto-layout-context';

export const DockviewTabCanvasViewer = memo((props: IDockviewPanelHeaderProps<DockviewPanelParameters>) => {
  const { t } = useTranslation();
  const isGenerationInProgress = useIsGenerationInProgress();
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

  return (
    <Flex ref={ref} position="relative" alignItems="center" h="full" onPointerDown={onPointerDown}>
      <Text userSelect="none" px={4}>
        {t(props.params.i18nKey)}
      </Text>
      {currentQueueItemDestination === 'canvas' && isGenerationInProgress && (
        <ProgressBar position="absolute" bottom={0} left={0} right={0} h={1} borderRadius="none" />
      )}
    </Flex>
  );
});
DockviewTabCanvasViewer.displayName = 'DockviewTabCanvasViewer';
