import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, forwardRef, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { ContainerElementSettings } from 'features/nodes/components/sidePanel/builder/ContainerElementSettings';
import { useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { NodeFieldElementSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementSettings';
import { formElementRemoved } from 'features/nodes/store/workflowSlice';
import { type FormElement, isContainerElement, isNodeFieldElement } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import { memo, useCallback } from 'react';
import { PiXBold } from 'react-icons/pi';

const getHeaderLabel = (el: FormElement) => {
  if (isContainerElement(el)) {
    if (el.data.layout === 'column') {
      return 'Column';
    }
    return 'Row';
  }
  return startCase(el.type);
};

const sx: SystemStyleObject = {
  w: 'full',
  ps: 2,
  h: 8,
  minH: 8,
  maxH: 8,
  borderTopRadius: 'inherit',
  borderColor: 'inherit',
  alignItems: 'center',
  cursor: 'grab',
  bg: 'base.700',
  '&[data-depth="0"]': { bg: 'base.800' },
  '&[data-depth="1"]': { bg: 'base.800' },
  '&[data-depth="2"]': { bg: 'base.750' },
};

export const FormElementEditModeHeader = memo(
  forwardRef(({ element }: { element: FormElement }, ref) => {
    const depth = useDepthContext();
    const dispatch = useAppDispatch();
    const removeElement = useCallback(() => {
      dispatch(formElementRemoved({ id: element.id }));
    }, [dispatch, element.id]);

    return (
      <Flex ref={ref} sx={sx} data-depth={depth}>
        <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
          {getHeaderLabel(element)} ({element.id})
        </Text>
        <Spacer />
        {isContainerElement(element) && <ContainerElementSettings element={element} />}
        {isNodeFieldElement(element) && <NodeFieldElementSettings element={element} />}
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
    );
  })
);
FormElementEditModeHeader.displayName = 'FormElementEditModeHeader';
