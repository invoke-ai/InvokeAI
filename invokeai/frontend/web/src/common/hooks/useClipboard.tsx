/* eslint-disable no-restricted-properties */

import { ExternalLink, Text } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { Param0 } from 'tsafe';

const CLIPBOARD_FAQ_URL = 'https://invoke-ai.github.io/InvokeAI/faq/#unable-to-copy-on-firefox';

export const useClipboard = () => {
  const { t } = useTranslation();
  const alertClipboardNotAvailable = useCallback(() => {
    toast({
      id: 'CLIPBOARD_UNAVAILABLE',
      title: t('toast.unableToCopy'),
      description: (
        <>
          <Text fontSize="md">
            {t('toast.unableToCopyDesc')}
            <ExternalLink
              display="inline"
              fontWeight="semibold"
              href={CLIPBOARD_FAQ_URL}
              label={t('toast.unableToCopyDesc_theseSteps')}
            />
            .
          </Text>
        </>
      ),
      status: 'error',
    });
  }, [t]);

  const isAvailable = useMemo(() => {
    if (!navigator.clipboard || !window.ClipboardItem) {
      return false;
    }
    // TODO(psyche): Should we query the permissions API?
    return true;
  }, []);

  const writeText = useCallback(
    (data: Param0<Clipboard['writeText']>, onCopy?: () => void) => {
      if (!isAvailable) {
        alertClipboardNotAvailable();
        return;
      }
      navigator.clipboard.writeText(data);
      onCopy?.();
    },
    [alertClipboardNotAvailable, isAvailable]
  );

  const write = useCallback(
    (data: Param0<Clipboard['write']>, onCopy?: () => void) => {
      if (!isAvailable) {
        alertClipboardNotAvailable();
        return;
      }
      navigator.clipboard.write(data);
      onCopy?.();
    },
    [alertClipboardNotAvailable, isAvailable]
  );

  const writeImage = useCallback(
    (blob: Blob, onCopy?: () => void) => {
      if (!isAvailable) {
        alertClipboardNotAvailable();
        return;
      }
      const data = [new ClipboardItem({ ['image/png']: blob })];
      navigator.clipboard.write(data);
      onCopy?.();
    },
    [alertClipboardNotAvailable, isAvailable]
  );

  return { isAvailable, writeText, write, writeImage };
};
