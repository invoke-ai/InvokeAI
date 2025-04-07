import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { ContainerElementSettings } from 'features/nodes/components/sidePanel/builder/ContainerElementSettings';
import { useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { NodeFieldElementSettings } from 'features/nodes/components/sidePanel/builder/NodeFieldElementSettings';
import { useMouseOverFormField } from 'features/nodes/hooks/useMouseOverNode';
import { useZoomToNode } from 'features/nodes/hooks/useZoomToNode';
import { formElementRemoved } from 'features/nodes/store/nodesSlice';
import type { FormElement, NodeFieldElement } from 'features/nodes/types/workflow';
import { isContainerElement, isNodeFieldElement } from 'features/nodes/types/workflow';
import { camelCase } from 'lodash-es';
import type { RefObject } from 'react';
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
  cursor: 'grab',
  '&[data-depth="0"]': { bg: 'baseAlpha.100' },
  '&[data-depth="1"]': { bg: 'baseAlpha.150' },
  '&[data-depth="2"]': { bg: 'baseAlpha.200' },
  '&[data-is-dragging="true"]': {
    opacity: 0.3,
  },
  _hover: {
    bg: 'baseAlpha.300',
  },
};

type Props = Omit<FlexProps, 'sx'> & { element: FormElement; dragHandleRef: RefObject<HTMLDivElement> };

export const FormElementEditModeHeader = memo(({ element, dragHandleRef, ...rest }: Props) => {
  const depth = useDepthContext();

  return (
    <Flex ref={dragHandleRef} sx={sx} data-depth={depth} {...rest}>
      <Label element={element} />
      <Spacer />
      {isContainerElement(element) && <ContainerElementSettings element={element} />}
      {isNodeFieldElement(element) && (
        <InputFieldGate
          nodeId={element.data.fieldIdentifier.nodeId}
          fieldName={element.data.fieldIdentifier.fieldName}
          fallback={null} // Do not render these buttons if the field is not found
        >
          <ZoomToNodeButton element={element} />
          <NodeFieldElementSettings element={element} />
        </InputFieldGate>
      )}
      <RemoveElementButton element={element} />
    </Flex>
  );
});
FormElementEditModeHeader.displayName = 'FormElementEditModeHeader';

const ZoomToNodeButton = memo(({ element }: { element: NodeFieldElement }) => {
  const { t } = useTranslation();
  const { nodeId } = element.data.fieldIdentifier;
  const zoomToNode = useZoomToNode(nodeId);
  const mouseOverFormField = useMouseOverFormField(nodeId);

  return (
    <IconButton
      onMouseOver={mouseOverFormField.handleMouseOver}
      onMouseOut={mouseOverFormField.handleMouseOut}
      tooltip={t('workflows.builder.zoomToNode')}
      aria-label={t('workflows.builder.zoomToNode')}
      onClick={zoomToNode}
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
  const { t } = useTranslation();
  const label = useMemo(() => {
    if (isContainerElement(element) && element.data.layout === 'column') {
      return t('workflows.builder.containerColumnLayout');
    }
    if (isContainerElement(element) && element.data.layout === 'row') {
      return t('workflows.builder.containerRowLayout');
    }
    return t(`workflows.builder.${camelCase(element.type)}`);
  }, [element, t]);

  return (
    <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all" userSelect="none">
      {label}
    </Text>
  );
});
Label.displayName = 'Label';
