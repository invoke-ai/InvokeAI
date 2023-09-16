import { ExternalLinkIcon } from '@chakra-ui/icons';
import { Flex, IconButton, Link, Text, Tooltip } from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCopy } from 'react-icons/fa';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';

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
const ImageMetadataItem = ({
  label,
  value,
  onClick,
  isLink,
  labelPosition,
  withCopy = false,
}: MetadataItemProps) => {
  const { t } = useTranslation();

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
            icon={<FaCopy />}
            size="xs"
            variant="ghost"
            fontSize={14}
            onClick={() => navigator.clipboard.writeText(value.toString())}
          />
        </Tooltip>
      )}
      <Flex direction={labelPosition ? 'column' : 'row'}>
        <Text fontWeight="semibold" whiteSpace="pre-wrap" pr={2}>
          {label}:
        </Text>
        {isLink ? (
          <Link href={value.toString()} isExternal wordBreak="break-all">
            {value.toString()} <ExternalLinkIcon mx="2px" />
          </Link>
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
