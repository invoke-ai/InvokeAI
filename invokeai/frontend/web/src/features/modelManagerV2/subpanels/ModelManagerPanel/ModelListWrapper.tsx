import { StickyScrollable } from 'features/system/components/StickyScrollable';
import type { AnyModelConfig } from 'services/api/types';

import ModelListItem from './ModelListItem';

type ModelListWrapperProps = {
  title: string;
  modelList: AnyModelConfig[];
};

export const ModelListWrapper = (props: ModelListWrapperProps) => {
  const { title, modelList } = props;
  return (
    <StickyScrollable title={title}>
      {modelList.map((model) => (
        <ModelListItem key={model.key} model={model} />
      ))}
    </StickyScrollable>
  );
};
