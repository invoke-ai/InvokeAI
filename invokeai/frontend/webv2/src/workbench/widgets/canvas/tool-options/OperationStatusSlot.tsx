import type { CSSProperties } from 'react';

import { Flex, IconButton, Spinner } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui';
import { InfoIcon } from 'lucide-react';

const ERROR_CLAMP_STYLE: CSSProperties = {
  display: '-webkit-box',
  overflow: 'hidden',
  WebkitBoxOrient: 'vertical',
  WebkitLineClamp: 2,
};

/**
 * The always-mounted status slot shared by the canvas operation bars: reserves
 * its width so status/error text appearing never shifts the surrounding
 * controls, and keeps the polite live region in the tree before content
 * arrives so announcements are reliable.
 */
export const OperationStatusSlot = ({
  errorDetail,
  errorText,
  isBusy,
  minW = '8rem',
  statusText,
  technicalDetailsLabel,
}: {
  errorDetail: string | null;
  errorText: string | null;
  isBusy: boolean;
  minW?: string;
  statusText: string;
  technicalDetailsLabel: string;
}) => {
  const detail = errorDetail?.trim();
  return (
    <Flex
      align="center"
      color={errorText ? 'fg.error' : 'fg.muted'}
      flex="0 1 auto"
      fontSize="xs"
      gap="1"
      maxW="16rem"
      minW={minW}
    >
      {errorText ? (
        <>
          <span aria-live="assertive" role="alert" style={ERROR_CLAMP_STYLE}>
            {errorText}
          </span>
          {detail && detail !== errorText ? (
            <Tooltip content={detail}>
              <IconButton aria-label={technicalDetailsLabel} flexShrink="0" size="xs" tabIndex={0} variant="ghost">
                <InfoIcon />
              </IconButton>
            </Tooltip>
          ) : null}
        </>
      ) : (
        <Flex align="center" aria-live="polite" gap="2" minW="0" role="status">
          {isBusy ? (
            <>
              <Spinner flexShrink="0" size="xs" />
              <span>{statusText}</span>
            </>
          ) : null}
        </Flex>
      )}
    </Flex>
  );
};
