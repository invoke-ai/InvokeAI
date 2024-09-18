import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, Collapse, Flex, Icon, Spacer, Text } from '@invoke-ai/ui-library';
import { useBoolean } from 'common/hooks/useBoolean';
import { CanvasEntityAddOfTypeButton } from 'features/controlLayers/components/common/CanvasEntityAddOfTypeButton';
import { CanvasEntityMergeVisibleButton } from 'features/controlLayers/components/common/CanvasEntityMergeVisibleButton';
import { CanvasEntityTypeIsHiddenToggle } from 'features/controlLayers/components/common/CanvasEntityTypeIsHiddenToggle';
import { useEntityTypeTitle } from 'features/controlLayers/hooks/useEntityTypeTitle';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';
import { PiCaretDownBold } from 'react-icons/pi';

type Props = PropsWithChildren<{
  isSelected: boolean;
  type: CanvasEntityIdentifier['type'];
}>;

const _hover: SystemStyleObject = {
  opacity: 1,
};

export const CanvasEntityGroupList = memo(({ isSelected, type, children }: Props) => {
  const title = useEntityTypeTitle(type);
  const collapse = useBoolean(true);
  const canMergeVisible = useMemo(() => type === 'raster_layer' || type === 'inpaint_mask', [type]);
  const canHideAll = useMemo(() => type !== 'reference_image', [type]);

  return (
    <Flex flexDir="column" w="full">
      <Flex w="full">
        <Flex
          flexGrow={1}
          as={Button}
          onClick={collapse.toggle}
          justifyContent="space-between"
          alignItems="center"
          gap={3}
          variant="unstyled"
          p={0}
          h={8}
        >
          <Icon
            boxSize={4}
            as={PiCaretDownBold}
            transform={collapse.isTrue ? undefined : 'rotate(-90deg)'}
            fill={isSelected ? 'base.200' : 'base.500'}
            transitionProperty="common"
            transitionDuration="fast"
          />
          <Text
            fontWeight="semibold"
            color={isSelected ? 'base.200' : 'base.500'}
            userSelect="none"
            transitionProperty="common"
            transitionDuration="fast"
          >
            {title}
          </Text>
          <Spacer />
        </Flex>
        {canMergeVisible && <CanvasEntityMergeVisibleButton type={type} />}
        {canHideAll && <CanvasEntityTypeIsHiddenToggle type={type} />}
        <CanvasEntityAddOfTypeButton type={type} />
      </Flex>
      <Collapse in={collapse.isTrue}>
        <Flex flexDir="column" gap={2} pt={2}>
          {children}
        </Flex>
      </Collapse>
    </Flex>
  );
});

CanvasEntityGroupList.displayName = 'CanvasEntityGroupList';
