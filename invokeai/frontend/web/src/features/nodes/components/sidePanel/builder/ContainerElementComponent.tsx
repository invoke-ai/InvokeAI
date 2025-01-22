import { Flex, type SystemStyleObject } from '@invoke-ai/ui-library';
import { ContainerContext } from 'features/nodes/components/sidePanel/builder/ContainerContext';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import { useElement } from 'features/nodes/store/workflowSlice';
import {
  CONTAINER_CLASS_NAME,
  DIVIDER_CLASS_NAME,
  isContainerElement,
  isDividerElement,
  isHeadingElement,
  isNodeFieldElement,
  isTextElement,
} from 'features/nodes/types/workflow';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const sx: SystemStyleObject = {
  gap: 4,
  '&[data-container-direction="column"]': {
    flexDir: 'column',
    '> :last-child': {
      flex: '1 0 0',
      alignItems: 'flex-start',
    },
  },
  '&[data-container-direction="row"]': {
    '> *': {
      flex: '1 1 0',
    },
  },
  [`& > .${DIVIDER_CLASS_NAME}`]: {
    flex: '0 0 1px',
  },
};

export const ContainerElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);

  if (!el || !isContainerElement(el)) {
    return null;
  }

  const { children, direction } = el.data;

  return (
    <ContainerContext.Provider value={el.data}>
      <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-direction={direction}>
        {children.map((childId) => (
          <FormElementComponent key={childId} id={childId} />
        ))}
      </Flex>
    </ContainerContext.Provider>
  );
});
ContainerElementComponent.displayName = 'ContainerElementComponent';

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
