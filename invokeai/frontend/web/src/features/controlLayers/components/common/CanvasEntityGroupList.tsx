import { Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { CanvasEntityTypeIsHiddenToggle } from 'features/controlLayers/components/common/CanvasEntityTypeIsHiddenToggle';
import { useEntityTypeTitle } from 'features/controlLayers/hooks/useEntityTypeTitle';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  isSelected: boolean;
  type: CanvasEntityIdentifier['type'];
}>;

export const CanvasEntityGroupList = memo(({ isSelected, type, children }: Props) => {
  const title = useEntityTypeTitle(type);
  return (
    <Flex flexDir="column" gap={2}>
      <Flex justifyContent="space-between" alignItems="center" gap={3}>
        <Text color={isSelected ? 'base.200' : 'base.500'} fontWeight="semibold" userSelect="none">
          {title}
        </Text>
        <Spacer />
        {type !== 'ip_adapter' && <CanvasEntityTypeIsHiddenToggle type={type} />}
      </Flex>
      {children}
    </Flex>
  );
});

CanvasEntityGroupList.displayName = 'CanvasEntityGroupList';
