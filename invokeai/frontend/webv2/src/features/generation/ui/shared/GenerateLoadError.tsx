import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { TriangleAlertIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const ERROR_ICON = <Icon as={TriangleAlertIcon} />;

interface GenerateLoadErrorProps {
  /** What could not be read, worded for the user. */
  message: string;
  /** The server's reason, when it gave one. */
  detail?: string | null;
  isRetrying: boolean;
  /** `state` fills a list that has nothing to show; `inline` sits above or below content that is still shown. */
  layout: 'inline' | 'state';
  onRetry: () => void;
}

/** A read that failed, never shown as an empty list: what failed, why when known, and a retry. */
export const GenerateLoadError = ({ detail, isRetrying, layout, message, onRetry }: GenerateLoadErrorProps) => {
  const { t } = useTranslation();
  const retry = (
    <Button flexShrink={0} loading={isRetrying} size="xs" variant="outline" onClick={onRetry}>
      {t('common.retry')}
    </Button>
  );

  if (layout === 'state') {
    return (
      <EmptyState danger description={detail} icon={ERROR_ICON} role="alert" title={message}>
        {retry}
      </EmptyState>
    );
  }

  return (
    <HStack alignItems="flex-start" gap="1.5" role="alert">
      <Icon as={TriangleAlertIcon} boxSize="3" color="fg.error" flexShrink={0} mt="px" />
      <Stack flex="1" gap="0.5" minW="0">
        <Text color="fg.error" fontSize="xs" textWrap="pretty">
          {message}
        </Text>
        {detail ? (
          <Text color="fg.muted" fontSize="xs" wordBreak="break-word">
            {detail}
          </Text>
        ) : null}
      </Stack>
      {retry}
    </HStack>
  );
};
