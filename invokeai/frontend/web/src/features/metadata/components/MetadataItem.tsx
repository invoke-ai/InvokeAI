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
  /** Display mode for the metadata item */
  displayMode?: 'default' | 'badge' | 'simple' | 'card';
  /** Color scheme for badge display mode */
  colorScheme?: string;
  /** Whether to show copy functionality */
  showCopy?: boolean;
  /** Whether to show recall functionality */
  showRecall?: boolean;
};

const _MetadataItem = typedMemo(<T,>({ 
  metadata, 
  handlers, 
  direction = 'row',
  displayMode = 'default',
  colorScheme = 'invokeBlue',
  showCopy = false,
  showRecall = true
}: MetadataItemProps<T>) => {
  const { label, isDisabled, value, renderedValue, onRecall, valueOrNull } = useMetadataItem(metadata, handlers);

  if (value === MetadataParseFailedToken) {
    return null;
  }

  if (handlers.getIsVisible && !isSymbol(value) && !handlers.getIsVisible(value)) {
    return null;
  }

  // For display modes other than default, we need the raw value for copy functionality
  if (displayMode !== 'default') {
    if (!valueOrNull) {
      return null;
    }
  }

  return (
    <MetadataItemView
      label={label}
      onRecall={showRecall ? onRecall : undefined}
      isDisabled={isDisabled}
      renderedValue={renderedValue}
      direction={direction}
      displayMode={displayMode}
      colorScheme={colorScheme}
      showCopy={showCopy}
      valueOrNull={valueOrNull}
    />
  );
});

export const MetadataItem = typedMemo(_MetadataItem);

MetadataItem.displayName = 'MetadataItem';
