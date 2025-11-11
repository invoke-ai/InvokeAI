import { Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { setFocusedRegion } from 'common/hooks/focus';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import type { TabName } from 'features/controlLayers/store/types';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type { IconType } from 'react-icons';
import {
  PiBoundingBoxBold,
  PiCubeBold,
  PiFlowArrowBold,
  PiFrameCornersBold,
  PiQueueBold,
  PiTextAaBold,
} from 'react-icons/pi';

import type { DockviewPanelParameters } from './auto-layout-context';

const TAB_ICONS: Record<TabName, IconType> = {
  generate: PiTextAaBold,
  canvas: PiBoundingBoxBold,
  upscaling: PiFrameCornersBold,
  workflows: PiFlowArrowBold,
  models: PiCubeBold,
  queue: PiQueueBold,
};

export const DockviewTabLaunchpad = memo((props: IDockviewPanelHeaderProps<DockviewPanelParameters>) => {
  const { t } = useTranslation();
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
      <Text userSelect="none">{t(props.params.i18nKey)}</Text>
    </Flex>
  );
});
DockviewTabLaunchpad.displayName = 'DockviewTabLaunchpad';
