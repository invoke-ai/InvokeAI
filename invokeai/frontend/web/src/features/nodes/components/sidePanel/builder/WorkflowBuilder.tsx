import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Alert, AlertDescription, AlertIcon, Button, ButtonGroup, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { firefoxDndFix } from 'features/dnd/util';
import { FormElementComponent } from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import {
  buildFormElementDndData,
  useBuilderDndMonitor,
  useRootDnd,
} from 'features/nodes/components/sidePanel/builder/dnd';
import { getEditModeWrapperId } from 'features/nodes/components/sidePanel/builder/shared';
import { formReset, selectFormIsEmpty, selectFormLayout } from 'features/nodes/store/workflowSlice';
import type { FormElement } from 'features/nodes/types/workflow';
import { buildContainer, buildDivider, buildHeading, buildText } from 'features/nodes/types/workflow';
import { startCase } from 'lodash-es';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { assert } from 'tsafe';

export const WorkflowBuilder = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEmpty = useAppSelector(selectFormIsEmpty);
  useBuilderDndMonitor();

  const resetForm = useCallback(() => {
    dispatch(formReset());
  }, [dispatch]);

  return (
    <ScrollableContent>
      <Flex justifyContent="center" w="full" h="full">
        <Flex flexDir="column" w="full" h="full" maxW="768px" gap={4}>
          <Alert status="warning" variant="subtle" borderRadius="base" flexShrink={0}>
            <AlertIcon />
            <AlertDescription fontSize="sm">{t('workflows.builder.workflowBuilderAlphaWarning')}</AlertDescription>
          </Alert>
          <ButtonGroup isAttached={false} justifyContent="center">
            <AddFormElementDndButton type="container" />
            <AddFormElementDndButton type="divider" />
            <AddFormElementDndButton type="heading" />
            <AddFormElementDndButton type="text" />
            <Spacer />
            <Button onClick={resetForm} variant="ghost" leftIcon={<PiArrowCounterClockwiseBold />}>
              {t('common.reset')}
            </Button>
          </ButtonGroup>
          {!isEmpty && <FormLayout />}
          {isEmpty && <EmptyStateEditMode />}
        </Flex>
      </Flex>
    </ScrollableContent>
  );
});
WorkflowBuilder.displayName = 'WorkflowBuilder';

export const FormLayout = memo(() => {
  const layout = useAppSelector(selectFormLayout);

  return (
    <Flex flexDir="column" gap={4} w="full" borderRadius="base">
      {layout.map((id) => (
        <FormElementComponent key={id} id={id} />
      ))}
    </Flex>
  );
});
FormLayout.displayName = 'FormLayout';

const EmptyStateEditMode = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const isDragging = useRootDnd(ref);

  return (
    <Flex
      id={getEditModeWrapperId('root')}
      ref={ref}
      w="full"
      h="full"
      bg={isDragging ? 'base.800' : undefined}
      p={4}
      borderRadius="base"
      alignItems="center"
      justifyContent="center"
    >
      <Text variant="subtext" fontSize="md">
        {t('workflows.builder.emptyRootPlaceholderEditMode')}
      </Text>
    </Flex>
  );
});
EmptyStateEditMode.displayName = 'EmptyStateEditMode';

const useAddFormElementDnd = (
  type: Exclude<FormElement['type'], 'node-field'>,
  draggableRef: RefObject<HTMLElement>
) => {
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const draggableElement = draggableRef.current;
    if (!draggableElement) {
      return;
    }
    return combine(
      firefoxDndFix(draggableElement),
      draggable({
        element: draggableElement,
        getInitialData: () => {
          if (type === 'container') {
            const element = buildContainer('row', []);
            return buildFormElementDndData(element);
          }
          if (type === 'divider') {
            const element = buildDivider();
            return buildFormElementDndData(element);
          }
          if (type === 'heading') {
            const element = buildHeading('');
            return buildFormElementDndData(element);
          }
          if (type === 'text') {
            const element = buildText('');
            return buildFormElementDndData(element);
          }
          assert(false);
        },
        onDragStart: () => {
          setIsDragging(true);
        },
        onDrop: () => {
          setIsDragging(false);
        },
      })
    );
  }, [draggableRef, type]);

  return isDragging;
};

const AddFormElementDndButton = ({ type }: { type: Parameters<typeof useAddFormElementDnd>[0] }) => {
  const draggableRef = useRef<HTMLDivElement>(null);
  const isDragging = useAddFormElementDnd(type, draggableRef);

  return (
    <Button
      as="div"
      ref={draggableRef}
      pointerEvents="all"
      variant="unstyled"
      borderWidth={2}
      borderStyle="dashed"
      borderRadius="base"
      px={4}
      py={1}
      cursor="grab"
      _hover={{ bg: 'base.800' }}
      isDisabled={isDragging}
    >
      {startCase(type)}
    </Button>
  );
};
