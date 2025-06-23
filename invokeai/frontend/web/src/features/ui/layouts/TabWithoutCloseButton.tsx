import { Flex, Text } from '@invoke-ai/ui-library';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import type { IDockviewPanelHeaderProps } from 'dockview';
import { useCallback, useRef } from 'react';

export const TabWithoutCloseButton = (props: IDockviewPanelHeaderProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const setActive = useCallback(() => {
    if (!props.api.isActive) {
      props.api.setActive();
    }
  }, [props.api]);

  useCallbackOnDragEnter(setActive, ref, 300);

  return (
    <Flex ref={ref} alignItems="center" h="full">
      <Text userSelect="none" px={4}>
        {props.api.title ?? props.api.id}
      </Text>
    </Flex>
  );
};
TabWithoutCloseButton.displayName = 'TabWithoutCloseButton';
