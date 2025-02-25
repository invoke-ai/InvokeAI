import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  ContainerContextProvider,
  DepthContextProvider,
  useDepthContext,
} from 'features/nodes/components/sidePanel/builder/contexts';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { useIsRootElement, useRootElementDropTarget } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import { selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import type { ContainerElement } from 'features/nodes/types/workflow';
import {
  CONTAINER_CLASS_NAME,
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const ContainerElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);
  const isRootElement = useIsRootElement(id);

  if (!el || !isContainerElement(el)) {
    return null;
  }

  if (isRootElement && mode === 'view') {
    return <RootContainerElementComponentViewMode el={el} />;
  }

  if (isRootElement && mode === 'edit') {
    return <RootContainerElementComponentEditMode el={el} />;
  }

  if (mode === 'view') {
    return <ContainerElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <ContainerElementComponentEditMode el={el} />;
});
ContainerElementComponent.displayName = 'ContainerElementComponent';

const rootViewModeSx: SystemStyleObject = {
  position: 'relative',
  alignItems: 'center',
  borderRadius: 'base',
  w: 'full',
  h: 'full',
  gap: 4,
  display: 'flex',
  flex: 1,
  '&[data-container-layout="column"]': {
    flexDir: 'column',
    alignItems: 'flex-start',
  },
  '&[data-container-layout="row"]': {
    flexDir: 'row',
  },
};

const RootContainerElementComponentViewMode = memo(({ el }: { el: ContainerElement }) => {
  const { id, data } = el;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={0}>
      <ContainerContextProvider id={id} layout={layout}>
        <Box id={id} className={CONTAINER_CLASS_NAME} sx={rootViewModeSx} data-container-layout={layout} data-depth={0}>
          {children.map((childId) => (
            <FormElementComponent key={childId} id={childId} />
          ))}
        </Box>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
RootContainerElementComponentViewMode.displayName = 'RootContainerElementComponentViewMode';

const rootEditModeSx: SystemStyleObject = {
  ...rootViewModeSx,
  '&[data-is-dragging-over="true"]': {
    opacity: 1,
    bg: 'base.850',
  },
};

const RootContainerElementComponentEditMode = memo(({ el }: { el: ContainerElement }) => {
  const { id, data } = el;
  const { children, layout } = data;
  const ref = useRef<HTMLDivElement>(null);
  const isDraggingOver = useRootElementDropTarget(ref);

  return (
    <DepthContextProvider depth={0}>
      <ContainerContextProvider id={id} layout={layout}>
        <Flex
          ref={ref}
          id={id}
          className={CONTAINER_CLASS_NAME}
          sx={rootEditModeSx}
          data-container-layout={layout}
          data-depth={0}
          data-is-dragging-over={isDraggingOver}
        >
          {children.map((childId) => (
            <FormElementComponent key={childId} id={childId} />
          ))}
          {children.length === 0 && <RootPlaceholder />}
        </Flex>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
RootContainerElementComponentEditMode.displayName = 'RootContainerElementComponentEditMode';

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

const ContainerElementComponentViewMode = memo(({ el }: { el: ContainerElement }) => {
  const { t } = useTranslation();
  const depth = useDepthContext();
  const { id, data } = el;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={depth + 1}>
      <ContainerContextProvider id={id} layout={layout}>
        <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-layout={layout} data-depth={depth}>
          {children.map((childId) => (
            <FormElementComponent key={childId} id={childId} />
          ))}
          {children.length === 0 && (
            <Flex p={8} w="full" h="full" alignItems="center" justifyContent="center">
              <Text variant="subtext">{t('workflows.builder.containerPlaceholder')}</Text>
            </Flex>
          )}
        </Flex>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
ContainerElementComponentViewMode.displayName = 'ContainerElementComponentViewMode';

const ContainerElementComponentEditMode = memo(({ el }: { el: ContainerElement }) => {
  const depth = useDepthContext();
  const { id, data } = el;
  const { children, layout } = data;

  return (
    <FormElementEditModeWrapper element={el}>
      <DepthContextProvider depth={depth + 1}>
        <ContainerContextProvider id={id} layout={layout}>
          <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-layout={layout} data-depth={depth}>
            {children.map((childId) => (
              <FormElementComponent key={childId} id={childId} />
            ))}
            {children.length === 0 && <NonRootPlaceholder />}
          </Flex>
        </ContainerContextProvider>
      </DepthContextProvider>
    </FormElementEditModeWrapper>
  );
});
ContainerElementComponentEditMode.displayName = 'ContainerElementComponentEditMode';

const RootPlaceholder = memo(() => {
  const { t } = useTranslation();
  return (
    <Flex p={8} w="full" h="full" alignItems="center" justifyContent="center">
      <Text variant="subtext">{t('workflows.builder.emptyRootPlaceholderEditMode')}</Text>
    </Flex>
  );
});
RootPlaceholder.displayName = 'RootPlaceholder';

const NonRootPlaceholder = memo(() => {
  const { t } = useTranslation();
  return (
    <Flex p={8} w="full" h="full" alignItems="center" justifyContent="center">
      <Text variant="subtext">{t('workflows.builder.containerPlaceholder')}</Text>
    </Flex>
  );
});
NonRootPlaceholder.displayName = 'NonRootPlaceholder';

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
