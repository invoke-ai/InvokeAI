import { autoScrollForElements } from '@atlaskit/pragmatic-drag-and-drop-auto-scroll/element';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  ContainerContextProvider,
  DepthContextProvider,
  useContainerContext,
  useDepthContext,
} from 'features/nodes/components/sidePanel/builder/contexts';
import { DividerElement } from 'features/nodes/components/sidePanel/builder/DividerElement';
import { useFormElementDnd, useRootElementDropTarget } from 'features/nodes/components/sidePanel/builder/dnd-hooks';
import { DndListDropIndicator } from 'features/nodes/components/sidePanel/builder/DndListDropIndicator';
import { FormElementEditModeContent } from 'features/nodes/components/sidePanel/builder/FormElementEditModeContent';
import { FormElementEditModeHeader } from 'features/nodes/components/sidePanel/builder/FormElementEditModeHeader';
import { HeadingElement } from 'features/nodes/components/sidePanel/builder/HeadingElement';
import { NodeFieldElement } from 'features/nodes/components/sidePanel/builder/NodeFieldElement';
import { TextElement } from 'features/nodes/components/sidePanel/builder/TextElement';
import { selectFormRootElement, selectWorkflowMode, useElement } from 'features/nodes/store/workflowSlice';
import type { ContainerElement } from 'features/nodes/types/workflow';
import {
  CONTAINER_CLASS_NAME,
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
  ROOT_CONTAINER_CLASS_NAME,
} from 'features/nodes/types/workflow';
import { memo, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const ContainerElement = memo(({ id }: { id: string }) => {
  const el = useElement(id);
  const mode = useAppSelector(selectWorkflowMode);

  if (!el || !isContainerElement(el)) {
    return null;
  }

  if (mode === 'view') {
    return <ContainerElementComponentViewMode el={el} />;
  }

  // mode === 'edit'
  return <ContainerElementComponentEditMode el={el} />;
});
ContainerElement.displayName = 'ContainerElementComponent';

const containerViewModeSx: SystemStyleObject = {
  gap: 2,
  '&[data-self-layout="column"]': {
    flexDir: 'column',
    alignItems: 'stretch',
  },
  '&[data-self-layout="row"]': {
    flexDir: 'row',
    alignItems: 'flex-start',
    overflowX: 'auto',
    overflowY: 'visible',
    h: 'min-content',
    flexShrink: 0,
  },
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
    minW: 32,
  },
};

const ContainerElementComponentViewMode = memo(({ el }: { el: ContainerElement }) => {
  const { t } = useTranslation();
  const depth = useDepthContext();
  const containerCtx = useContainerContext();
  const { id, data } = el;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={depth + 1}>
      <ContainerContextProvider id={id} layout={layout}>
        <Flex
          id={id}
          className={CONTAINER_CLASS_NAME}
          sx={containerViewModeSx}
          data-self-layout={layout}
          data-depth={depth}
          data-parent-layout={containerCtx.layout}
        >
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

const containerEditModeSx: SystemStyleObject = {
  borderRadius: 'base',
  position: 'relative',
  '&[data-active-drop-region="center"]': {
    opacity: 1,
    bg: 'base.850',
  },
  flexDir: 'column',
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
    h: 'min-content',
  },
};

const containerEditModeContentSx: SystemStyleObject = {
  gap: 4,
  p: 4,
  flex: '1 1 0',
  '&[data-self-layout="column"]': {
    flexDir: 'column',
  },
  '&[data-self-layout="row"]': {
    flexDir: 'row',
    overflowX: 'auto',
  },
};

const ContainerElementComponentEditMode = memo(({ el }: { el: ContainerElement }) => {
  const depth = useDepthContext();
  const draggableRef = useRef<HTMLDivElement>(null);
  const dragHandleRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef<HTMLDivElement>(null);
  const [activeDropRegion, isDragging] = useFormElementDnd(el.id, draggableRef, dragHandleRef);
  const { id, data } = el;
  const { children, layout } = data;
  const containerCtx = useContainerContext();

  useEffect(() => {
    const element = autoScrollRef.current;
    if (!element) {
      return;
    }

    if (layout === 'column') {
      // No need to auto-scroll for column layout
      return;
    }

    return autoScrollForElements({
      element,
    });
  }, [layout]);

  return (
    <DepthContextProvider depth={depth + 1}>
      <ContainerContextProvider id={id} layout={layout}>
        <Flex
          id={id}
          ref={draggableRef}
          className={CONTAINER_CLASS_NAME}
          sx={containerEditModeSx}
          data-depth={depth}
          data-parent-layout={containerCtx.layout}
          data-active-drop-region={activeDropRegion}
        >
          <FormElementEditModeHeader dragHandleRef={dragHandleRef} element={el} data-is-dragging={isDragging} />
          <FormElementEditModeContent data-is-dragging={isDragging}>
            <Flex ref={autoScrollRef} sx={containerEditModeContentSx} data-self-layout={layout}>
              {children.map((childId) => (
                <FormElementComponent key={childId} id={childId} />
              ))}
              {children.length === 0 && <NonRootPlaceholder />}
            </Flex>
          </FormElementEditModeContent>
          <DndListDropIndicator activeDropRegion={activeDropRegion} gap="var(--invoke-space-4)" />
        </Flex>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
ContainerElementComponentEditMode.displayName = 'ContainerElementComponentEditMode';

const rootViewModeSx: SystemStyleObject = {
  position: 'relative',
  alignItems: 'center',
  borderRadius: 'base',
  w: 'full',
  h: 'full',
  gap: 2,
  display: 'flex',
  flex: 1,
  maxW: '768px',
  '&[data-self-layout="column"]': {
    flexDir: 'column',
    alignItems: 'stretch',
  },
  '&[data-self-layout="row"]': {
    flexDir: 'row',
    alignItems: 'flex-start',
  },
};

export const RootContainerElementViewMode = memo(() => {
  const el = useAppSelector(selectFormRootElement);
  const { id, data } = el;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={0}>
      <ContainerContextProvider id={id} layout={layout}>
        <Box id={id} className={ROOT_CONTAINER_CLASS_NAME} sx={rootViewModeSx} data-self-layout={layout} data-depth={0}>
          {children.map((childId) => (
            <FormElementComponent key={childId} id={childId} />
          ))}
        </Box>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
RootContainerElementViewMode.displayName = 'RootContainerElementViewMode';

const rootEditModeSx: SystemStyleObject = {
  ...rootViewModeSx,
  gap: 4,
  '&[data-is-dragging-over="true"]': {
    opacity: 1,
    bg: 'base.850',
  },
};

export const RootContainerElementEditMode = memo(() => {
  const el = useAppSelector(selectFormRootElement);
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
          className={ROOT_CONTAINER_CLASS_NAME}
          sx={rootEditModeSx}
          data-self-layout={layout}
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
RootContainerElementEditMode.displayName = 'RootContainerElementEditMode';

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
const FormElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);

  if (!el) {
    return null;
  }

  if (isContainerElement(el)) {
    return <ContainerElement key={id} id={id} />;
  }

  if (isNodeFieldElement(el)) {
    return <NodeFieldElement key={id} id={id} />;
  }

  if (isDividerElement(el)) {
    return <DividerElement key={id} id={id} />;
  }

  if (isHeadingElement(el)) {
    return <HeadingElement key={id} id={id} />;
  }

  if (isTextElement(el)) {
    return <TextElement key={id} id={id} />;
  }

  assert<Equals<typeof el, never>>(false, `Unhandled type for element with id ${id}`);
});
FormElementComponent.displayName = 'FormElementComponent';
