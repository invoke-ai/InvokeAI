import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import { Button, Collapse, Flex, Icon, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useBoolean } from 'common/hooks/useBoolean';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { fixTooltipCloseOnScrollStyles } from 'common/util/fixTooltipCloseOnScrollStyles';
import { CanvasEntityAddOfTypeButton } from 'features/controlLayers/components/common/CanvasEntityAddOfTypeButton';
import { CanvasEntityMergeVisibleButton } from 'features/controlLayers/components/common/CanvasEntityMergeVisibleButton';
import { CanvasEntityTypeIsHiddenToggle } from 'features/controlLayers/components/common/CanvasEntityTypeIsHiddenToggle';
import { useEntityTypeInformationalPopover } from 'features/controlLayers/hooks/useEntityTypeInformationalPopover';
import { useEntityTypeTitle } from 'features/controlLayers/hooks/useEntityTypeTitle';
import { entitiesReordered } from 'features/controlLayers/store/canvasSlice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isRenderableEntityType } from 'features/controlLayers/store/types';
import { singleCanvasEntityDndSource } from 'features/dnd/dnd';
import { triggerPostMoveFlash } from 'features/dnd/util';
import type { PropsWithChildren } from 'react';
import { memo, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { PiCaretDownBold } from 'react-icons/pi';

type Props = PropsWithChildren<{
  isSelected: boolean;
  type: CanvasEntityIdentifier['type'];
  entityIdentifiers: CanvasEntityIdentifier[];
}>;

export const CanvasEntityGroupList = memo(({ isSelected, type, children, entityIdentifiers }: Props) => {
  const title = useEntityTypeTitle(type);
  const informationalPopoverFeature = useEntityTypeInformationalPopover(type);
  const collapse = useBoolean(true);
  const dispatch = useAppDispatch();

  useEffect(() => {
    return monitorForElements({
      canMonitor({ source }) {
        if (!singleCanvasEntityDndSource.typeGuard(source.data)) {
          return false;
        }
        if (source.data.payload.entityIdentifier.type !== type) {
          return false;
        }
        return true;
      },
      onDrop({ location, source }) {
        const target = location.current.dropTargets[0];
        if (!target) {
          return;
        }

        const sourceData = source.data;
        const targetData = target.data;

        if (!singleCanvasEntityDndSource.typeGuard(sourceData) || !singleCanvasEntityDndSource.typeGuard(targetData)) {
          return;
        }

        const indexOfSource = entityIdentifiers.findIndex(
          (entityIdentifier) => entityIdentifier.id === sourceData.payload.entityIdentifier.id
        );
        const indexOfTarget = entityIdentifiers.findIndex(
          (entityIdentifier) => entityIdentifier.id === targetData.payload.entityIdentifier.id
        );

        if (indexOfTarget < 0 || indexOfSource < 0) {
          return;
        }

        // Don't move if the source and target are the same index, meaning same position in the list
        if (indexOfSource === indexOfTarget) {
          return;
        }

        const closestEdgeOfTarget = extractClosestEdge(targetData);

        // It's possible that the indices are different, but refer to the same position. For example, if the source is
        // at 2 and the target is at 3, but the target edge is 'top', then the entity is already in the correct position.
        // We should bail if this is the case.
        let edgeIndexDelta = 0;

        if (closestEdgeOfTarget === 'bottom') {
          edgeIndexDelta = 1;
        } else if (closestEdgeOfTarget === 'top') {
          edgeIndexDelta = -1;
        }

        // If the source is already in the correct position, we don't need to move it.
        if (indexOfSource === indexOfTarget + edgeIndexDelta) {
          return;
        }

        // Using `flushSync` so we can query the DOM straight after this line
        flushSync(() => {
          dispatch(
            entitiesReordered({
              type,
              entityIdentifiers: reorderWithEdge({
                list: entityIdentifiers,
                startIndex: indexOfSource,
                indexOfTarget,
                closestEdgeOfTarget,
                axis: 'vertical',
              }),
            })
          );
        });

        // Flash the element that was moved
        const element = document.querySelector(`[data-entity-id="${sourceData.payload.entityIdentifier.id}"]`);
        if (element instanceof HTMLElement) {
          triggerPostMoveFlash(element, colorTokenToCssVar('base.700'));
        }
      },
    });
  }, [dispatch, entityIdentifiers, type]);

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
          {informationalPopoverFeature ? (
            <InformationalPopover feature={informationalPopoverFeature}>
              <Text
                fontWeight="semibold"
                color={isSelected ? 'base.200' : 'base.500'}
                userSelect="none"
                transitionProperty="common"
                transitionDuration="fast"
              >
                {title}
              </Text>
            </InformationalPopover>
          ) : (
            <Text
              fontWeight="semibold"
              color={isSelected ? 'base.200' : 'base.500'}
              userSelect="none"
              transitionProperty="common"
              transitionDuration="fast"
            >
              {title}
            </Text>
          )}

          <Spacer />
        </Flex>
        {isRenderableEntityType(type) && <CanvasEntityMergeVisibleButton type={type} />}
        {isRenderableEntityType(type) && <CanvasEntityTypeIsHiddenToggle type={type} />}
        <CanvasEntityAddOfTypeButton type={type} />
      </Flex>
      <Collapse in={collapse.isTrue} style={fixTooltipCloseOnScrollStyles}>
        <Flex flexDir="column" gap={2} pt={2}>
          {children}
        </Flex>
      </Collapse>
    </Flex>
  );
});

CanvasEntityGroupList.displayName = 'CanvasEntityGroupList';
