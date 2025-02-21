import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, forwardRef, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { ContainerElementSettings } from 'features/nodes/components/sidePanel/builder/ContainerElementSettings';
import { useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { NodeFieldElementSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementSettings';
import { useZoomToNode } from 'features/nodes/hooks/useZoomToNode';
import { formElementRemoved } from 'features/nodes/store/workflowSlice';
import type { FormElement, NodeFieldElement } from 'features/nodes/types/workflow';
import { isContainerElement, isNodeFieldElement } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGpsFixBold, PiXBold } from 'react-icons/pi';

const sx: SystemStyleObject = {
  w: 'full',
  ps: 2,
  h: 8,
  minH: 8,
  maxH: 8,
  borderTopRadius: 'base',
  alignItems: 'center',
  color: 'base.500',
  bg: 'baseAlpha.250',
  '&[data-depth="0"]': { bg: 'baseAlpha.100' },
  '&[data-depth="1"]': { bg: 'baseAlpha.150' },
  '&[data-depth="2"]': { bg: 'baseAlpha.200' },
};

export const FormElementEditModeHeader = memo(
  forwardRef(({ element }: { element: FormElement }, ref) => {
    const depth = useDepthContext();

    return (
      <Flex ref={ref} sx={sx} data-depth={depth}>
        <Label element={element} />
        <Spacer />
        {isContainerElement(element) && <ContainerElementSettings element={element} />}
        {isNodeFieldElement(element) && <ZoomToNodeButton element={element} />}
        {isNodeFieldElement(element) && <NodeFieldElementSettings element={element} />}
        <RemoveElementButton element={element} />
      </Flex>
    );
  })
);
FormElementEditModeHeader.displayName = 'FormElementEditModeHeader';

const ZoomToNodeButton = memo(({ element }: { element: NodeFieldElement }) => {
  const { t } = useTranslation();
  const zoomToNode = useZoomToNode();
  const onClick = useCallback(() => {
    zoomToNode(element.data.fieldIdentifier.nodeId);
  }, [element.data.fieldIdentifier.nodeId, zoomToNode]);

  return (
    <IconButton
      tooltip={t('workflows.builder.zoomToNode')}
      aria-label={t('workflows.builder.zoomToNode')}
      onClick={onClick}
      icon={<PiGpsFixBold />}
      variant="link"
      size="sm"
      alignSelf="stretch"
    />
  );
});
ZoomToNodeButton.displayName = 'ZoomToNodeButton';

const RemoveElementButton = memo(({ element }: { element: FormElement }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const removeElement = useCallback(() => {
    dispatch(formElementRemoved({ id: element.id }));
  }, [dispatch, element.id]);

  return (
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
  );
});
RemoveElementButton.displayName = 'RemoveElementButton';

const Label = memo(({ element }: { element: FormElement }) => {
  const label = useMemo(() => {
    if (isContainerElement(element) && element.data.layout === 'column') {
      return `Container (column layout)`;
    }
    if (isContainerElement(element) && element.data.layout === 'row') {
      return `Container (row layout)`;
    }
    return startCase(element.type);
  }, [element]);

  return (
    <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all" userSelect="none">
      {label}
    </Text>
  );
});
Label.displayName = 'Label';
