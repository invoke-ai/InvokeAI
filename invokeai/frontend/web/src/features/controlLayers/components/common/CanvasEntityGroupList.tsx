import { monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, Collapse, Flex, Icon, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useBoolean } from 'common/hooks/useBoolean';
import { fixTooltipCloseOnScrollStyles } from 'common/util/fixTooltipCloseOnScrollStyles';
import { CanvasEntityAddOfTypeButton } from 'features/controlLayers/components/common/CanvasEntityAddOfTypeButton';
import { CanvasEntityMergeVisibleButton } from 'features/controlLayers/components/common/CanvasEntityMergeVisibleButton';
import { CanvasEntityTypeIsHiddenToggle } from 'features/controlLayers/components/common/CanvasEntityTypeIsHiddenToggle';
import { useEntityTypeInformationalPopover } from 'features/controlLayers/hooks/useEntityTypeInformationalPopover';
import { useEntityTypeTitle } from 'features/controlLayers/hooks/useEntityTypeTitle';
import { entitiesReordered } from 'features/controlLayers/store/canvasSlice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { isRenderableEntityType } from 'features/controlLayers/store/types';
import { Dnd } from 'features/dnd/dnd';
import type { PropsWithChildren } from 'react';
import { memo, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { PiCaretDownBold } from 'react-icons/pi';

type Props = PropsWithChildren<{
  isSelected: boolean;
  type: CanvasEntityIdentifier['type'];
  entityIdentifiers: CanvasEntityIdentifier[];
}>;

const _hover: SystemStyleObject = {
  opacity: 1,
};

export const CanvasEntityGroupList = memo(({ isSelected, type, children, entityIdentifiers }: Props) => {
  const title = useEntityTypeTitle(type);
  const informationalPopoverFeature = useEntityTypeInformationalPopover(type);
  const collapse = useBoolean(true);
  const dispatch = useAppDispatch();

  useEffect(() => {
    return monitorForElements({
      canMonitor({ source }) {
        if (!Dnd.Source.singleCanvasEntity.typeGuard(source.data)) {
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

        if (
          !Dnd.Source.singleCanvasEntity.typeGuard(sourceData) ||
          !Dnd.Source.singleCanvasEntity.typeGuard(targetData)
        ) {
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

        const closestEdgeOfTarget = extractClosestEdge(targetData);

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
        // // Being simple and just querying for the task after the drop.
        // // We could use react context to register the element in a lookup,
        // // and then we could retrieve that element after the drop and use
        // // `triggerPostMoveFlash`. But this gets the job done.
        // const element = document.querySelector(`[data-task-id="${sourceData.taskId}"]`);
        // if (element instanceof HTMLElement) {
        //   triggerPostMoveFlash(element);
        // }
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
