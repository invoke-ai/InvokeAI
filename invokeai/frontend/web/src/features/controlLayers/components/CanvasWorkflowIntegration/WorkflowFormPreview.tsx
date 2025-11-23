import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import { WorkflowFieldRenderer } from 'features/controlLayers/components/CanvasWorkflowIntegration/WorkflowFieldRenderer';
import { selectCanvasWorkflowIntegrationSelectedWorkflowId } from 'features/controlLayers/store/canvasWorkflowIntegrationSlice';
import {
  ContainerContextProvider,
  DepthContextProvider,
  useContainerContext,
  useDepthContext,
} from 'features/nodes/components/sidePanel/builder/contexts';
import { DividerElement } from 'features/nodes/components/sidePanel/builder/DividerElement';
import { HeadingElement } from 'features/nodes/components/sidePanel/builder/HeadingElement';
import { TextElement } from 'features/nodes/components/sidePanel/builder/TextElement';
import type { FormElement } from 'features/nodes/types/workflow';
import {
  CONTAINER_CLASS_NAME,
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
  ROOT_CONTAINER_CLASS_NAME,
} from 'features/nodes/types/workflow';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';

const log = logger('canvas-workflow-integration');

const rootViewModeSx: SystemStyleObject = {
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

export const WorkflowFormPreview = memo(() => {
  const { t } = useTranslation();
  const selectedWorkflowId = useAppSelector(selectCanvasWorkflowIntegrationSelectedWorkflowId);

  const { data: workflow, isLoading } = useGetWorkflowQuery(selectedWorkflowId!, {
    skip: !selectedWorkflowId,
  });

  const elements = useMemo((): Record<string, FormElement> => {
    if (!workflow?.workflow.form) {
      return {};
    }
    const els = workflow.workflow.form.elements as Record<string, FormElement>;
    log.debug({ elementCount: Object.keys(els).length, elementIds: Object.keys(els) }, 'Form elements loaded');
    return els;
  }, [workflow]);

  const rootElementId = useMemo((): string => {
    if (!workflow?.workflow.form) {
      return '';
    }
    const rootId = workflow.workflow.form.rootElementId as string;
    log.debug({ rootElementId: rootId }, 'Root element ID');
    return rootId;
  }, [workflow]);

  if (isLoading) {
    return (
      <Flex alignItems="center" gap={2}>
        <Spinner size="sm" />
        <Text>{t('controlLayers.workflowIntegration.loadingParameters', 'Loading workflow parameters...')}</Text>
      </Flex>
    );
  }

  if (!workflow) {
    return null;
  }

  // If workflow has no form builder, it should have been filtered out
  // This is a fallback in case something went wrong
  if (Object.keys(elements).length === 0 || !rootElementId) {
    return (
      <Text fontSize="sm" color="error.400">
        {t(
          'controlLayers.workflowIntegration.noFormBuilderError',
          'This workflow has no form builder and cannot be used. Please select a different workflow.'
        )}
      </Text>
    );
  }

  const rootElement = elements[rootElementId];

  if (!rootElement || !isContainerElement(rootElement)) {
    return null;
  }

  const { id, data } = rootElement;
  const { children, layout } = data;

  return (
    <DepthContextProvider depth={0}>
      <ContainerContextProvider id={id} layout={layout}>
        <Box id={id} className={ROOT_CONTAINER_CLASS_NAME} sx={rootViewModeSx} data-self-layout={layout} data-depth={0}>
          {children.map((childId) => (
            <FormElementComponentPreview key={childId} id={childId} elements={elements} />
          ))}
        </Box>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
WorkflowFormPreview.displayName = 'WorkflowFormPreview';

const FormElementComponentPreview = memo(({ id, elements }: { id: string; elements: Record<string, FormElement> }) => {
  const el = elements[id];

  if (!el) {
    log.warn({ id }, 'Element not found in elements map');
    return null;
  }

  log.debug({ id, type: el.type }, 'Rendering form element');

  if (isContainerElement(el)) {
    return <ContainerElementPreview el={el} elements={elements} />;
  }

  if (isDividerElement(el)) {
    return <DividerElement id={id} />;
  }

  if (isHeadingElement(el)) {
    return <HeadingElement id={id} />;
  }

  if (isTextElement(el)) {
    return <TextElement id={id} />;
  }

  if (isNodeFieldElement(el)) {
    return <WorkflowFieldRenderer el={el} />;
  }

  // If we get here, it's an unknown element type
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  log.warn({ id, type: (el as any).type }, 'Unknown element type - not rendering');
  return null;
});
FormElementComponentPreview.displayName = 'FormElementComponentPreview';

const ContainerElementPreview = memo(({ el, elements }: { el: FormElement; elements: Record<string, FormElement> }) => {
  const { t } = useTranslation();
  const depth = useDepthContext();
  const containerCtx = useContainerContext();

  if (!isContainerElement(el)) {
    return null;
  }

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
            <FormElementComponentPreview key={childId} id={childId} elements={elements} />
          ))}
          {children.length === 0 && (
            <Flex p={8} w="full" h="full" alignItems="center" justifyContent="center">
              <Text color="base.500" fontSize="sm" fontStyle="oblique 10deg">
                {t('workflows.builder.emptyContainer')}
              </Text>
            </Flex>
          )}
        </Flex>
      </ContainerContextProvider>
    </DepthContextProvider>
  );
});
ContainerElementPreview.displayName = 'ContainerElementPreview';
