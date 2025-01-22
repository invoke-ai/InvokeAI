import { Flex, type SystemStyleObject } from '@invoke-ai/ui-library';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { ContainerContext } from 'features/nodes/components/sidePanel/builder/ContainerContext';
import {
  DIVIDER_CLASS_NAME,
  DividerElementComponent,
} from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import { useElement } from 'features/nodes/types/workflow';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const CONTAINER_CLASS_NAME = getPrefixedId('container');

const sx: SystemStyleObject = {
  gap: 4,
  '&[data-container-direction="column"]': {
    flexDir: 'column',
    flex: '1 1 0',
    // Select all non-divider children (dividers have a fixed width that they define on their own)
    [`> *:not(.${DIVIDER_CLASS_NAME})`]: {
      // By default, all children should take up the same amount of space
      flex: '0 1 0',
      // The last child should take up the remaining space
      '&:last-child': {
        flex: '1 1 auto',
      },
    },
  },
  '&[data-container-direction="row"]': {
    // Select all non-divider children (dividers have a fixed width that they define on their own)
    [`> *:not(.${DIVIDER_CLASS_NAME})`]: {
      // By default, all children should take up the same amount of space
      flex: '1 1 0',
    },
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
// export const ContainerElementComponent = memo(({ id }: { id: string }) => {
//   const element = useElement(id);

//   if (!element || element.type !== 'container') {
//     return null;
//   }

//   const { children, direction } = element.data;

//   return (
//     <GridItem
//       as={Grid}
//       id={id}
//       gap={4}
//       w="full"
//       h="full"
//       gridTemplateColumns={direction === 'row' ? fill(children.length, '1fr') : undefined}
//       gridTemplateRows={direction === 'column' ? fill(children.length, 'min-content', '1fr') : undefined}
//       gridAutoFlow={direction === 'column' ? 'row' : 'column'}
//       alignItems="baseline"
//     >
//       {children.map((childId) => (
//         <FormElementComponent key={childId} id={childId} />
//       ))}
//     </GridItem>
//   );
// });
// ContainerElementComponent.displayName = 'ContainerElementComponent';

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
