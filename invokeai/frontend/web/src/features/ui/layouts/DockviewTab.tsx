import { Flex, Text } from '@invoke-ai/ui-library';
import { setFocusedRegion } from 'common/hooks/focus';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { DockviewPanelParameters } from './auto-layout-context';

export const DockviewTab = memo((props: IDockviewPanelHeaderProps<DockviewPanelParameters>) => {
  const { t } = useTranslation();
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
    <Flex ref={ref} alignItems="center" h="full" onPointerDown={onPointerDown}>
      <Text userSelect="none" px={4}>
        {t(props.params.i18nKey)}
      </Text>
    </Flex>
  );
});
DockviewTab.displayName = 'DockviewTab';
