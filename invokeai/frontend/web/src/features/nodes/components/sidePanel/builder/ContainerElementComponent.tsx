import { ContainerDirectionContext } from 'features/nodes/components/sidePanel/builder/ContainerContext';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { ElementWrapper } from 'features/nodes/components/sidePanel/builder/ElementWrapper';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import { useElement } from 'features/nodes/types/workflow';
import { memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const getGridTemplateColumns = (count: number) => {
  return Array.from({ length: count }, () => '1fr').join(' ');
};
const fill = (count: number, val: string, last?: string) => {
  return Array.from({ length: count }, (_, i) => {
    if (last && i === count - 1) {
      return last;
    }
    return val;
  }).join(' ');
};

export const ContainerElementComponent = memo(({ id }: { id: string }) => {
  const element = useElement(id);

  if (!element || element.type !== 'container') {
    return null;
  }

  const { children, direction } = element.data;

  return (
    <ContainerDirectionContext.Provider value={direction}>
      <ElementWrapper id={id} gap={4} flexDir={direction}>
        {children.map((childId) => (
          <FormElementComponent key={childId} id={childId} />
        ))}
      </ElementWrapper>
    </ContainerDirectionContext.Provider>
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
