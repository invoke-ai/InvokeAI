import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { StickyScrollable } from 'features/system/components/StickyScrollable';
import { memo } from 'react';
import type { AnyModelConfig } from 'services/api/types';

import ModelListItem from './ModelListItem';

type ModelListWrapperProps = {
  title: string;
  modelList: AnyModelConfig[];
};

const headingSx = {
  bg: 'base.900',
  pb: 3,
  pl: 3,
} satisfies SystemStyleObject;

const contentSx = {
  gap: 0,
  p: 0,
  bg: 'base.900',
  borderRadius: '0',
} satisfies SystemStyleObject;

export const ModelListWrapper = memo((props: ModelListWrapperProps) => {
  const { title, modelList } = props;
  if (modelList.length === 0) {
    return null;
  }
  return (
    <StickyScrollable title={title} contentSx={contentSx} headingSx={headingSx}>
      {modelList.map((model) => (
        <ModelListItem key={model.key} model={model} />
      ))}
    </StickyScrollable>
  );
});

ModelListWrapper.displayName = 'ModelListWrapper';
