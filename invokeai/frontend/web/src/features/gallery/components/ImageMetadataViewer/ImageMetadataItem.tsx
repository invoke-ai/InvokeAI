import { ExternalLinkIcon } from '@chakra-ui/icons';
import { Flex, Link } from '@chakra-ui/react';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
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
const ImageMetadataItem = ({
  label,
  value,
  onClick,
  isLink,
  labelPosition,
  withCopy = false,
}: MetadataItemProps) => {
  const { t } = useTranslation();

  const handleCopy = useCallback(
    () => navigator.clipboard.writeText(value.toString()),
    [value]
  );

  if (!value) {
    return null;
  }

  return (
    <Flex gap={2}>
      {onClick && (
        <InvTooltip label={`Recall ${label}`}>
          <InvIconButton
            aria-label={t('accessibility.useThisParameter')}
            icon={<IoArrowUndoCircleOutline />}
            size="xs"
            variant="ghost"
            fontSize={20}
            onClick={onClick}
          />
        </InvTooltip>
      )}
      {withCopy && (
        <InvTooltip label={`Copy ${label}`}>
          <InvIconButton
            aria-label={`Copy ${label}`}
            icon={<PiCopyBold />}
            size="xs"
            variant="ghost"
            fontSize={14}
            onClick={handleCopy}
          />
        </InvTooltip>
      )}
      <Flex direction={labelPosition ? 'column' : 'row'}>
        <InvText fontWeight="semibold" whiteSpace="pre-wrap" pr={2}>
          {label}:
        </InvText>
        {isLink ? (
          <Link href={value.toString()} isExternal wordBreak="break-all">
            {value.toString()} <ExternalLinkIcon mx="2px" />
          </Link>
        ) : (
          <InvText overflowY="scroll" wordBreak="break-all">
            {value.toString()}
          </InvText>
        )}
      </Flex>
    </Flex>
  );
};

export default memo(ImageMetadataItem);
