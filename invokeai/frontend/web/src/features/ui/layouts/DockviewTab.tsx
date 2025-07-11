import { Flex, Text } from '@invoke-ai/ui-library';
import { setFocusedRegion } from 'common/hooks/focus';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { memo, useCallback, useRef } from 'react';

import type { PanelParameters } from './auto-layout-context';
import { useHackOutDvTabDraggable } from './use-hack-out-dv-tab-draggable';

export const DockviewTab = memo((props: IDockviewPanelHeaderProps<PanelParameters>) => {
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

  useHackOutDvTabDraggable(ref);

  return (
    <Flex ref={ref} alignItems="center" h="full" onPointerDown={onPointerDown}>
      <Text userSelect="none" px={4}>
        {props.api.title ?? props.api.id}
      </Text>
    </Flex>
  );
});
DockviewTab.displayName = 'DockviewTab';
