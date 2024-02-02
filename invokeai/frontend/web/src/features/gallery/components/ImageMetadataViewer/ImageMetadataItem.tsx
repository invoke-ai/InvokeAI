import { ExternalLink, Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';
import { PiCopyBold } from 'react-icons/pi';

type MetadataItemProps = {
  isLink?: boolean;
  label: string;
  onClick?: () => void;
  value: number | string | boolean;
  labelPosition?: string;
  withCopy?: boolean;
};

/**
 * Component to display an individual metadata item or parameter.
 */
const ImageMetadataItem = ({ label, value, onClick, isLink, labelPosition, withCopy = false }: MetadataItemProps) => {
  const { t } = useTranslation();

  const handleCopy = useCallback(() => navigator.clipboard.writeText(value.toString()), [value]);

  if (!value) {
    return null;
  }

  return (
    <Flex gap={2}>
      {onClick && (
        <Tooltip label={`Recall ${label}`}>
          <IconButton
            aria-label={t('accessibility.useThisParameter')}
            icon={<IoArrowUndoCircleOutline />}
            size="xs"
            variant="ghost"
            fontSize={20}
            onClick={onClick}
          />
        </Tooltip>
      )}
      {withCopy && (
        <Tooltip label={`Copy ${label}`}>
          <IconButton
            aria-label={`Copy ${label}`}
            icon={<PiCopyBold />}
            size="xs"
            variant="ghost"
            fontSize={14}
            onClick={handleCopy}
          />
        </Tooltip>
      )}
      <Flex direction={labelPosition ? 'column' : 'row'}>
        <Text fontWeight="semibold" whiteSpace="pre-wrap" pr={2}>
          {label}:
        </Text>
        {isLink ? (
          <ExternalLink href={value.toString()} label={value.toString()} />
        ) : (
          <Text overflowY="scroll" wordBreak="break-all">
            {value.toString()}
          </Text>
        )}
      </Flex>
    </Flex>
  );
};

export default memo(ImageMetadataItem);
