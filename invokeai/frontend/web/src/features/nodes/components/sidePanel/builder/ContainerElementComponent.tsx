import { Flex, type SystemStyleObject } from '@invoke-ai/ui-library';
import { ContainerContext } from 'features/nodes/components/sidePanel/builder/ContainerContext';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import { CONTAINER_CLASS_NAME, DIVIDER_CLASS_NAME, useElement } from 'features/nodes/types/workflow';
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
  const element = useElement(id);

  if (!element || element.type !== 'container') {
    return null;
  }

  const { children, direction } = element.data;

  return (
    <ContainerContext.Provider value={element.data}>
      <Flex id={id} className={CONTAINER_CLASS_NAME} sx={sx} data-container-direction={direction}>
        {children.map((childId) => (
          <FormElementComponent key={childId} id={childId} />
        ))}
      </Flex>
    </ContainerContext.Provider>
  );
});
ContainerElementComponent.displayName = 'ContainerElementComponent';

export const FormElementComponent = memo(({ id }: { id: string }) => {
  const element = useElement(id);

  if (!element) {
    return null;
  }

  const { type } = element;

  switch (type) {
    case 'container':
      return <ContainerElementComponent key={id} id={id} />;
    case 'node-field':
      return <NodeFieldElementComponent key={id} id={id} />;
    case 'divider':
      return <DividerElementComponent key={id} id={id} />;
    case 'heading':
      return <HeadingElementComponent key={id} id={id} />;
    case 'text':
      return <TextElementComponent key={id} id={id} />;
    default:
      assert<Equals<typeof type, never>>(false, `Unhandled type ${type}`);
  }
});
FormElementComponent.displayName = 'FormElementComponent';
