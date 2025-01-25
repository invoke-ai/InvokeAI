import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { useDraggableFormElement } from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import { formElementRemoved } from 'features/nodes/store/workflowSlice';
import { type FormElement, isContainerElement } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import { memo, useCallback, useRef } from 'react';
import { PiXBold } from 'react-icons/pi';

export const EDIT_MODE_WRAPPER_CLASS_NAME = getPrefixedId('edit-mode-wrapper', '-');

export const getEditModeWrapperId = (id: string) => `${id}-edit-mode-wrapper`;

const getHeaderLabel = (el: FormElement) => {
  if (isContainerElement(el)) {
    if (el.data.direction === 'column') {
      return 'Column';
    }
    return 'Row';
  }
  return startCase(el.type);
};

const wrapperSx: SystemStyleObject = {
  position: 'relative',
  flexDir: 'column',
  boxShadow: '0 0 0 1px var(--invoke-colors-base-750)',
  borderRadius: 'base',
  alignItems: 'center',
  justifyContent: 'flex-start',
  w: 'full',
  h: 'full',
  '&[data-is-dragging="true"]': {
    opacity: 0.3,
  },
  '&[data-active-drop-region="center"]': {
    opacity: 1,
    bg: 'base.700',
  },
};

const headerSx: SystemStyleObject = {
  w: 'full',
  ps: 2,
  h: 8,
  borderTopRadius: 'inherit',
  borderColor: 'inherit',
  alignItems: 'center',
  cursor: 'grab',
  bg: 'base.700',
  '&[data-depth="0"]': { bg: 'base.800' },
  '&[data-depth="1"]': { bg: 'base.800' },
  '&[data-depth="2"]': { bg: 'base.750' },
};

export const FormElementEditModeWrapper = memo(
  ({ element, children, ...rest }: { element: FormElement } & FlexProps) => {
    const draggableRef = useRef<HTMLDivElement>(null);
    const dragHandleRef = useRef<HTMLDivElement>(null);
    const [activeDropRegion, isDragging] = useDraggableFormElement(element.id, draggableRef, dragHandleRef);
    const depth = useDepthContext();
    const dispatch = useAppDispatch();
    const removeElement = useCallback(() => {
      dispatch(formElementRemoved({ id: element.id }));
    }, [dispatch, element.id]);

    return (
      <Flex
        id={getEditModeWrapperId(element.id)}
        ref={draggableRef}
        sx={wrapperSx}
        className={EDIT_MODE_WRAPPER_CLASS_NAME}
        data-is-dragging={isDragging}
        data-active-drop-region={activeDropRegion}
        {...rest}
      >
        <Flex ref={dragHandleRef} sx={headerSx} data-depth={depth}>
          <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
            {getHeaderLabel(element)} ({element.id})
          </Text>
          <Spacer />
          {element.parentId && (
            <IconButton
              aria-label="delete"
              onClick={removeElement}
              icon={<PiXBold />}
              variant="link"
              size="sm"
              alignSelf="stretch"
              colorScheme="error"
            />
          )}
        </Flex>
        <Flex w="full" p={4} alignItems="center" gap={4}>
          {children}
        </Flex>
        <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
      </Flex>
    );
  }
);

FormElementEditModeWrapper.displayName = 'FormElementEditModeWrapper';
