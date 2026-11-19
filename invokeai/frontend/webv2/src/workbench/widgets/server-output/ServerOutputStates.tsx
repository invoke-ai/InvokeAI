import { Center, HStack, Spinner, Text, VStack } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { LockIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ServerOutputRefusalMessage } from './serverOutputStatus';

const REFUSAL_ICON = <LockIcon />;

export const ServerOutputLoading = () => {
  const { t } = useTranslation();

  return (
    <Center flex="1" minH="0">
      <VStack color="fg.muted" gap="2">
        <Spinner />
        <Text fontSize="xs">{t('widgets.serverOutput.loading')}</Text>
      </VStack>
    </Center>
  );
};

/**
 * Discovery drops a refused widget, so its status-bar chip goes too: the refusal offers its own way out. Check again
 * reads the grant once more; it is offered only when the session's role could be granted at all.
 */
export const ServerOutputRefusal = ({
  isChecking,
  onCheckAgain,
  onRemove,
  reason,
}: {
  isChecking: boolean;
  onCheckAgain?: () => void;
  onRemove: () => void;
  reason: ServerOutputRefusalMessage;
}) => {
  const { t } = useTranslation();

  return (
    <Center flex="1" minH="0" p="4">
      <EmptyState
        description={t(`widgets.serverOutput.refused.${reason}`)}
        icon={REFUSAL_ICON}
        title={t('widgets.serverOutput.refused.title')}
      >
        <HStack gap="2" justify="center" wrap="wrap">
          {onCheckAgain ? (
            <Button loading={isChecking} size="sm" variant="outline" onClick={onCheckAgain}>
              {t('widgets.serverOutput.refused.checkAgain')}
            </Button>
          ) : null}
          <Button size="sm" variant="ghost" onClick={onRemove}>
            {t('widgets.serverOutput.refused.remove')}
          </Button>
        </HStack>
      </EmptyState>
    </Center>
  );
};
