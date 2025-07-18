import { Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { setFocusedRegion } from 'common/hooks/focus';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import { memo, useCallback, useRef } from 'react';
import type { IconType } from 'react-icons';
import {
  PiBoundingBoxBold,
  PiCubeBold,
  PiFlowArrowBold,
  PiFrameCornersBold,
  PiQueueBold,
  PiTextAaBold,
} from 'react-icons/pi';

const TAB_ICONS: Record<TabName, IconType> = {
  generate: PiTextAaBold,
  canvas: PiBoundingBoxBold,
  upscaling: PiFrameCornersBold,
  workflows: PiFlowArrowBold,
  models: PiCubeBold,
  queue: PiQueueBold,
};

export const DockviewTabLaunchpad = memo((props: IDockviewPanelHeaderProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const activeTab = useAppSelector(selectActiveTab);

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
    <Flex ref={ref} alignItems="center" h="full" px={4} gap={3} onPointerDown={onPointerDown}>
      <Icon as={TAB_ICONS[activeTab]} color="invokeYellow.300" boxSize={5} />
      <Text userSelect="none">{props.api.title ?? props.api.id}</Text>
    </Flex>
  );
});
DockviewTabLaunchpad.displayName = 'DockviewTabLaunchpad';
