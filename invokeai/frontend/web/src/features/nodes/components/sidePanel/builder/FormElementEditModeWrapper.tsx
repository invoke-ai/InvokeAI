import { Flex, type FlexProps, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import type { DndListTargetState } from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import { useDraggableFormElement } from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import { formElementRemoved } from 'features/nodes/store/workflowSlice';
import { type FormElement, isContainerElement } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import { memo, useCallback, useRef } from 'react';
import { PiXBold } from 'react-icons/pi';

export const EDIT_MODE_WRAPPER_CLASS_NAME = getPrefixedId('edit-mode-wrapper', '-');

export const getEditModeWrapperId = (id: string) => `${id}-edit-mode-wrapper`;

const getHeaderBgColor = (depth: number) => {
  if (depth <= 1) {
    return 'base.800';
  }
  if (depth === 2) {
    return 'base.750';
  }
  return 'base.700';
};

const getHeaderLabel = (el: FormElement) => {
  if (isContainerElement(el)) {
    if (el.data.direction === 'column') {
      return 'Column';
    }
    return 'Row';
  }
  return startCase(el.type);
};

const getBgColor = (dndListState: DndListTargetState) => {
  switch (dndListState.type) {
    case 'idle':
      return undefined;
    case 'is-dragging':
      return 'red';
    case 'is-dragging-over':
      if (dndListState.closestCenterOrEdge === 'center') {
        return 'magenta';
      }
      return 'blue';
    case 'preview':
      return 'green';
  }
};

export const FormElementEditModeWrapper = memo(
  ({ element, children, ...rest }: { element: FormElement } & FlexProps) => {
    const draggableRef = useRef<HTMLDivElement>(null);
    const dragHandleRef = useRef<HTMLDivElement>(null);
    const [dndListState] = useDraggableFormElement(element.id, draggableRef, dragHandleRef);
    const depth = useDepthContext();
    const dispatch = useAppDispatch();
    const removeElement = useCallback(() => {
      dispatch(formElementRemoved({ id: element.id }));
    }, [dispatch, element.id]);

    if (dndListState.type !== 'idle') {
      // console.log(element.id, 'dndListState', dndListState);
    }

    return (
      <Flex
        id={getEditModeWrapperId(element.id)}
        ref={draggableRef}
        position="relative"
        className={EDIT_MODE_WRAPPER_CLASS_NAME}
        flexDir="column"
        boxShadow="0 0 0 1px var(--invoke-colors-base-750)"
        borderRadius="base"
        alignItems="center"
        justifyContent="flex-start"
        w="full"
        h="full"
        bg={getBgColor(dndListState)}
        opacity={dndListState.type === 'is-dragging' ? 0.3 : 1}
        {...rest}
      >
        <Flex
          ref={dragHandleRef}
          w="full"
          ps={2}
          h={8}
          bg={getHeaderBgColor(depth)}
          borderTopRadius="inherit"
          // borderBottomWidth={1}
          borderColor="inherit"
          alignItems="center"
          cursor="grab"
        >
          <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
            {element.id}
            {/* {getHeaderLabel(element)} */}
          </Text>
          <Spacer />
          <IconButton
            aria-label="delete"
            onClick={removeElement}
            icon={<PiXBold />}
            variant="link"
            size="sm"
            alignSelf="stretch"
            colorScheme="error"
          />
        </Flex>
        <Flex w="full" p={4} alignItems="center" gap={4}>
          {children}
        </Flex>
        <DndListDropIndicator dndState={dndListState} gap="var(--invoke-space-4)" />
      </Flex>
    );
  }
);

FormElementEditModeWrapper.displayName = 'FormElementEditModeWrapper';
