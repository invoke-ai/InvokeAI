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
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

const getHeaderLabel = (el: FormElement) => {
  if (isContainerElement(el)) {
    if (el.data.layout === 'column') {
      return 'Container (column layout)';
    }
    return 'Container (row layout)';
  }
  return startCase(el.type);
};

const sx: SystemStyleObject = {
  w: 'full',
  ps: 2,
  h: 8,
  minH: 8,
  maxH: 8,
  borderTopRadius: 'base',
  borderColor: 'base.800',
  alignItems: 'center',
  cursor: 'grab',
  color: 'base.300',
  borderBottomWidth: 1,
  bg: 'base.700',
  '&[data-depth="0"]': { bg: 'base.800' },
  '&[data-depth="1"]': { bg: 'base.750' },
  '&[data-depth="2"]': { bg: 'base.700' },
};

export const FormElementEditModeHeader = memo(
  forwardRef(({ element }: { element: FormElement }, ref) => {
    const { t } = useTranslation();
    const depth = useDepthContext();
    const dispatch = useAppDispatch();
    const removeElement = useCallback(() => {
      dispatch(formElementRemoved({ id: element.id }));
    }, [dispatch, element.id]);

    return (
      <Flex ref={ref} sx={sx} data-depth={depth}>
        <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
          {getHeaderLabel(element)}
        </Text>
        <Spacer />
        {isContainerElement(element) && <ContainerElementSettings element={element} />}
        {isNodeFieldElement(element) && <NodeFieldElementSettings element={element} />}
        <IconButton
          tooltip={t('common.delete')}
          aria-label={t('common.delete')}
          onClick={removeElement}
          icon={<PiXBold />}
          variant="link"
          size="sm"
          alignSelf="stretch"
          colorScheme="error"
        />
      </Flex>
    );
  })
);
FormElementEditModeHeader.displayName = 'FormElementEditModeHeader';
