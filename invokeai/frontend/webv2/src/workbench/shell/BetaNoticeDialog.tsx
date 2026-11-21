import { Link, Portal, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { patchWorkbenchPreferences, useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const ISSUES_URL = 'https://github.com/invoke-ai/InvokeAI/issues';

const acknowledge = () => void patchWorkbenchPreferences({ betaNoticeAcknowledged: true });

/** Wait for account preferences before showing the notice; dismissal follows the account. */
export const BetaNoticeDialog = ({ onExitComplete }: { onExitComplete?: () => void }) => {
  const { t } = useTranslation();
  const isOpen = useWorkbenchSettingsSelector(
    (snapshot) => snapshot.status === 'ready' && !snapshot.preferences.betaNoticeAcknowledged
  );
  const dismissRef = useRef<HTMLButtonElement | null>(null);
  const handleOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      acknowledge();
    }
  }, []);
  // The dismiss button is the primary action; the issues link stays reachable by Tab.
  const getInitialFocusEl = useCallback(() => dismissRef.current, []);

  return (
    <Dialog.Root
      initialFocusEl={getInitialFocusEl}
      open={isOpen}
      role="alertdialog"
      size="sm"
      onExitComplete={onExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{t('betaNotice.title')}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="2">
                <Text fontSize="md">{t('betaNotice.body')}</Text>
                <Text fontSize="md">
                  {t('betaNotice.reportPrefix')}{' '}
                  <Link color="accent.fg" href={ISSUES_URL} rel="noreferrer" target="_blank">
                    {t('betaNotice.reportLink')}
                  </Link>
                  .
                </Text>
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button ref={dismissRef} colorPalette="accent" variant="solid" onClick={acknowledge}>
                {t('betaNotice.dismiss')}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
