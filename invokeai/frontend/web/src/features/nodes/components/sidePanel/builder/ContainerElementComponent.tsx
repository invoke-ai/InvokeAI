import { Flex, IconButton, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ContainerContext, DepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { FormElementEditModeWrapper } from 'features/nodes/components/sidePanel/builder/FormElementEditModeWrapper';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import { useMonitorForFormElementDnd } from 'features/nodes/components/sidePanel/builder/use-builder-dnd';
import { formElementAdded, selectWorkflowFormMode, useElement } from 'features/nodes/store/workflowSlice';
import type { ContainerElement } from 'features/nodes/types/workflow';
import {
  container,
  CONTAINER_CLASS_NAME,
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { memo, useCallback, useContext } from 'react';
import { PiPlusBold } from 'react-icons/pi';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const sx: SystemStyleObject = {
  gap: 4,
  flex: '1 1 0',
  '&[data-container-direction="column"]': {
    flexDir: 'column',
  },
  '&[data-container-direction="row"]': {
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
  const depth = useContext(DepthContext);
  const { id, data } = el;
  const { children, direction } = data;

  return (
    <DepthContext.Provider value={depth + 1}>
      <ContainerContext.Provider value={data}>
        <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-direction={direction}>
          {children.map((childId) => (
            <FormElementComponent key={childId} id={childId} />
          ))}
        </Flex>
      </ContainerContext.Provider>{' '}
    </DepthContext.Provider>
  );
});
ContainerElementComponentViewMode.displayName = 'ContainerElementComponentViewMode';

export const ContainerElementComponentEditMode = memo(({ el }: { el: ContainerElement }) => {
  const depth = useContext(DepthContext);
  const { id, data } = el;
  const { children, direction } = data;
  useMonitorForFormElementDnd(id, children);

  return (
    <FormElementEditModeWrapper element={el}>
      <DepthContext.Provider value={depth + 1}>
        <ContainerContext.Provider value={data}>
          <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-direction={direction}>
            {children.map((childId) => (
              <FormElementComponent key={childId} id={childId} />
            ))}
            {direction === 'row' && children.length < 3 && depth < 2 && <AddColumnButton containerId={id} />}
            {direction === 'column' && depth < 1 && <AddRowButton containerId={id} />}
          </Flex>
        </ContainerContext.Provider>
      </DepthContext.Provider>
    </FormElementEditModeWrapper>
  );
});
ContainerElementComponentEditMode.displayName = 'ContainerElementComponentEditMode';

const AddColumnButton = ({ containerId }: { containerId: string }) => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    const element = container('column', []);
    dispatch(formElementAdded({ element, containerId }));
  }, [containerId, dispatch]);
  return (
    <IconButton onClick={onClick} aria-label="add column" icon={<PiPlusBold />} h="unset" variant="ghost" size="sm" />
  );
};

const AddRowButton = ({ containerId }: { containerId: string }) => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    const element = container('row', []);
    dispatch(formElementAdded({ element, containerId }));
  }, [containerId, dispatch]);
  return (
    <IconButton onClick={onClick} aria-label="add row" icon={<PiPlusBold />} w="unset" variant="ghost" size="sm" />
  );
};

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
