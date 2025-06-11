import { Flex, Text } from '@invoke-ai/ui-library';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { useCallback, useEffect, useId, useRef } from 'react';

export const TabWithoutCloseButton = (props: IDockviewPanelHeaderProps) => {
  const id = useId();
  const ref = useRef<HTMLDivElement>(null);
  const setActive = useCallback(() => {
    if (!props.api.isActive) {
      props.api.setActive();
    }
  }, [props.api]);

  useCallbackOnDragEnter(setActive, ref, 300);

  useEffect(() => {
    const el = document.querySelector(`[data-id="${id}"]`);
    if (!el) {
      return;
    }
    const parentTab = el.closest('.dv-tab');
    if (!parentTab) {
      return;
    }
    parentTab.setAttribute('draggable', 'false');
  }, [id]);

  return (
    <Flex ref={ref}>
      <Text userSelect="none">{props.api.title ?? props.api.id}</Text>
    </Flex>
  );
};
TabWithoutCloseButton.displayName = 'TabWithoutCloseButton';
