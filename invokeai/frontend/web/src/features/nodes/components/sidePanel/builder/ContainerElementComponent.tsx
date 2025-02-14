import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  ContainerContextProvider,
  DepthContextProvider,
  useDepthContext,
} from 'features/nodes/components/sidePanel/builder/contexts';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import { selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { ContainerElement } from 'features/nodes/types/workflow';
import {
  CONTAINER_CLASS_NAME,
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const sx: SystemStyleObject = {
  gap: 4,
  flex: '1 1 0',
  '&[data-container-layout="column"]': {
    flexDir: 'column',
  },
  '&[data-container-layout="row"]': {
    flexDir: 'row',
  },
};

export const ContainerElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowFormMode);

  if (!el || !isContainerElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <ContainerElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <ContainerElementComponentEditMode el={el} />;
});
ContainerElementComponent.displayName = 'ContainerElementComponent';

export const ContainerElementComponentViewMode = memo(({ el }: { el: ContainerElement }) => {
  const depth = useDepthContext();
  const { id, data } = el;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={depth + 1}>
      <ContainerContextProvider id={id} layout={layout}>
        <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-layout={layout}>
          {children.map((childId) => (
            <FormElementComponent key={childId} id={childId} />
          ))}
        </Flex>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
ContainerElementComponentViewMode.displayName = 'ContainerElementComponentViewMode';

export const ContainerElementComponentEditMode = memo(({ el }: { el: ContainerElement }) => {
  const { t } = useTranslation();
  const depth = useDepthContext();
  const { id, data } = el;
  const { children, layout } = data;

  return (
    <FormElementEditModeWrapper element={el}>
      <DepthContextProvider depth={depth + 1}>
        <ContainerContextProvider id={id} layout={layout}>
          <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-layout={layout}>
            {children.map((childId) => (
              <FormElementComponent key={childId} id={childId} />
            ))}
            {children.length === 0 && (
              <Flex p={4} w="full" h="full" alignItems="center" justifyContent="center">
                <Text variant="subtext">{t('workflows.builder.emptyContainerPlaceholder')}</Text>
              </Flex>
            )}
          </Flex>
        </ContainerContextProvider>
      </DepthContextProvider>
    </FormElementEditModeWrapper>
  );
});
ContainerElementComponentEditMode.displayName = 'ContainerElementComponentEditMode';

// TODO(psyche): Can we move this into a separate file and avoid circular dependencies between it and ContainerElementComponent?
export const FormElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);

  if (!el) {
    return null;
  }

  if (isContainerElement(el)) {
    return <ContainerElementComponent key={id} id={id} />;
  }

  if (isNodeFieldElement(el)) {
    return <NodeFieldElementComponent key={id} id={id} />;
  }

  if (isDividerElement(el)) {
    return <DividerElementComponent key={id} id={id} />;
  }

  if (isHeadingElement(el)) {
    return <HeadingElementComponent key={id} id={id} />;
  }

  if (isTextElement(el)) {
    return <TextElementComponent key={id} id={id} />;
  }

  assert<Equals<typeof el, never>>(false, `Unhandled type for element with id ${id}`);
});
FormElementComponent.displayName = 'FormElementComponent';
