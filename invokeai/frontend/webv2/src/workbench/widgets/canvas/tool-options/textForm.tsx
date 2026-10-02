import type { ToolPropertyForm } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
import type { ComponentType, ReactNode } from 'react';

import { Skeleton } from '@chakra-ui/react';
import { lazy, Suspense } from 'react';

const loadingRows = <Skeleton height="8" />;
// Matches the specimen card so the form does not jump when the chunk lands.
const loadingCard = <Skeleton height="20" />;

// Catalog queries and variable-font controls are needed only while text is selected.
const lazyPart = <Props extends object>(
  load: () => Promise<{ default: ComponentType<Props> }>,
  fallback: ReactNode = loadingRows
) => {
  const Body = lazy(load);
  return (props: Props) => (
    <Suspense fallback={fallback}>
      <Body {...props} />
    </Suspense>
  );
};

export const textForm: ToolPropertyForm = {
  groups: [
    {
      body: lazyPart(() => import('./TextOptions').then((module) => ({ default: module.TextFontSettings }))),
      id: 'text-font',
      labelKey: 'widgets.properties.groups.font',
    },
    {
      body: lazyPart(() => import('./TextOptions').then((module) => ({ default: module.TextParagraphSettings }))),
      id: 'text-paragraph',
      labelKey: 'widgets.properties.groups.paragraph',
    },
    {
      body: lazyPart(() => import('./TextOptions').then((module) => ({ default: module.TextColorSettings }))),
      id: 'text-color',
      labelKey: 'widgets.properties.rows.color',
    },
  ],
  id: 'text',
  paintsLeaf: true,
  preview: lazyPart(() => import('./TextOptions').then((module) => ({ default: module.TextPreview })), loadingCard),
};
