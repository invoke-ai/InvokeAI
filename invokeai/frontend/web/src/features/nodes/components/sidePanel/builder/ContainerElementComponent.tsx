import { Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import type { ContainerElement, FormElement } from 'features/nodes/types/workflow';
import { Fragment, memo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const getGridTemplateColumns = (count: number) => {
  return Array.from({ length: count }, () => '1fr').join(' auto ');
};

export const ContainerElementComponent = memo(({ element }: { element: ContainerElement }) => {
  const { id, data } = element;
  const { columns } = data;

  return (
    <Grid id={id} gap={4} gridTemplateColumns={getGridTemplateColumns(columns.length)} gridAutoFlow="column">
      {columns.map((elements, columnIndex) => {
        const key = `${element.id}_${columnIndex}`;
        const withDivider = columnIndex < columns.length - 1;
        return (
          <Fragment key={key}>
            <GridItem as={Grid} id={key} gap={4} gridAutoRows="min-content" gridAutoFlow="row">
              {elements.map((element) => (
                <FormElementComponent key={element.id} element={element} />
              ))}
            </GridItem>
            {withDivider && <Flex w="1px" bg="base.800" flexShrink={0} />}
          </Fragment>
        );
      })}
    </Grid>
  );
});
ContainerElementComponent.displayName = 'ContainerElementComponent';

export const FormElementComponent = memo(({ element }: { element: FormElement }) => {
  const { type, id } = element;
  switch (type) {
    case 'container':
      return <ContainerElementComponent key={id} element={element} />;
    case 'node-field':
      return <NodeFieldElementComponent key={id} element={element} />;
    case 'divider':
      return <DividerElementComponent key={id} element={element} />;
    case 'heading':
      return <HeadingElementComponent key={id} element={element} />;
    case 'text':
      return <TextElementComponent key={id} element={element} />;
    default:
      assert<Equals<typeof type, never>>(false, `Unhandled type ${type}`);
  }
});
FormElementComponent.displayName = 'FormElementComponent';
