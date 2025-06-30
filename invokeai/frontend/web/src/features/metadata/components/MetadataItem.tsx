import { typedMemo } from '@invoke-ai/ui-library';
import { isSymbol } from 'es-toolkit/compat';
import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import { useMetadataItem } from 'features/metadata/hooks/useMetadataItem';
import type { MetadataHandlers } from 'features/metadata/types';
import { MetadataParseFailedToken } from 'features/metadata/util/parsers';

type MetadataItemProps<T> = {
  metadata: unknown;
  handlers: MetadataHandlers<T>;
  direction?: 'row' | 'column';
};

const _MetadataItem = typedMemo(<T,>({ metadata, handlers, direction = 'row' }: MetadataItemProps<T>) => {
  const { label, isDisabled, value, renderedValue, onRecall } = useMetadataItem(metadata, handlers);

  if (value === MetadataParseFailedToken) {
    return null;
  }

  if (handlers.getIsVisible && !isSymbol(value) && !handlers.getIsVisible(value)) {
    return null;
  }

  return (
    <MetadataItemView
      label={label}
      onRecall={onRecall}
      isDisabled={isDisabled}
      renderedValue={renderedValue}
      direction={direction}
    />
  );
});

export const MetadataItem = typedMemo(_MetadataItem);

MetadataItem.displayName = 'MetadataItem';
