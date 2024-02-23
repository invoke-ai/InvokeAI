import { Flex, Text } from '@invoke-ai/ui-library';
import { RecallButton } from 'features/metadata/components/RecallButton';
import { memo } from 'react';

type MetadataItemViewProps = {
  onRecall: () => void;
  label: string;
  renderedValue: React.ReactNode;
  isDisabled: boolean;
  direction?: 'row' | 'column';
};

export const MetadataItemView = memo(
  ({ label, onRecall, isDisabled, renderedValue, direction = 'row' }: MetadataItemViewProps) => {
    return (
      <Flex gap={2}>
        {onRecall && <RecallButton label={label} onClick={onRecall} isDisabled={isDisabled} />}
        <Flex direction={direction}>
          <Text fontWeight="semibold" whiteSpace="pre-wrap" pr={2}>
            {label}:
          </Text>
          {renderedValue}
        </Flex>
      </Flex>
    );
  }
);

MetadataItemView.displayName = 'MetadataItemView';
