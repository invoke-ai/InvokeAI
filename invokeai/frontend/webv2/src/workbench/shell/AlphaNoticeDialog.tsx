import { Dialog, Link, Portal, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui/Button';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { patchWorkbenchPreferences, useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const ISSUES_URL = 'https://github.com/invoke-ai/InvokeAI/issues';

const acknowledge = () => void patchWorkbenchPreferences({ alphaNoticeAcknowledged: true });

/** Mounted only while the notice is open: workbench hotkeys stay quiet under it, as under the other dialogs. */
const AlphaNoticeModalLayer = () => {
  useMountEffect(() => registerHotkeyModalLayer('alpha-notice'));

  return null;
};

/** Wait for account preferences before showing the notice; dismissal follows the account. */
export const AlphaNoticeDialog = () => {
  const { t } = useTranslation();
  const isOpen = useWorkbenchSettingsSelector(
    (snapshot) => snapshot.status === 'ready' && !snapshot.preferences.alphaNoticeAcknowledged
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
      onOpenChange={handleOpenChange}
    >
      {isOpen ? <AlphaNoticeModalLayer /> : null}
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{t('alphaNotice.title')}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="2">
                <Text fontSize="xs">{t('alphaNotice.body')}</Text>
                <Text fontSize="xs">
                  {t('alphaNotice.reportPrefix')}{' '}
                  <Link color="accent.fg" href={ISSUES_URL} rel="noreferrer" target="_blank">
                    {t('alphaNotice.reportLink')}
                  </Link>
                  .
                </Text>
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button ref={dismissRef} colorPalette="accent" size="xs" variant="solid" onClick={acknowledge}>
                {t('alphaNotice.dismiss')}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
