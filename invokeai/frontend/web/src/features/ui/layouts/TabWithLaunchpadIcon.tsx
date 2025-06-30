import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import { memo, useCallback, useRef } from 'react';
import {
  PiBoundingBoxBold,
  PiCubeBold,
  PiFlowArrowBold,
  PiFrameCornersBold,
  PiQueueBold,
  PiTextAaBold,
} from 'react-icons/pi';

const TAB_ICONS: Record<TabName, React.ReactElement> = {
  generate: <PiTextAaBold />,
  canvas: <PiBoundingBoxBold />,
  upscaling: <PiFrameCornersBold />,
  workflows: <PiFlowArrowBold />,
  models: <PiCubeBold />,
  queue: <PiQueueBold />,
};

export const TabWithLaunchpadIcon = memo((props: IDockviewPanelHeaderProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const activeTab = useAppSelector(selectActiveTab);

  const setActive = useCallback(() => {
    if (!props.api.isActive) {
      props.api.setActive();
    }
  }, [props.api]);

  useCallbackOnDragEnter(setActive, ref, 300);

  // Show icon only for Launchpad panel
  const isLaunchpadPanel = props.api.id === 'launchpad';
  const currentTabIcon = TAB_ICONS[activeTab];

  return (
    <Flex ref={ref} alignItems="center" h="full">
      {isLaunchpadPanel && currentTabIcon && (
        <Flex alignItems="center" px={2}>
          {currentTabIcon}
        </Flex>
      )}
      <Text userSelect="none" px={isLaunchpadPanel ? 2 : 4}>
        {props.api.title ?? props.api.id}
      </Text>
    </Flex>
  );
});
TabWithLaunchpadIcon.displayName = 'TabWithLaunchpadIcon';
