import { ExternalLink, Text, useToast } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { atom } from 'nanostores';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const TOAST_ID = 'hfForbidden';

/**
 * Tracks whether or not the HF Login toast is showing
 */
export const $isHFForbiddenToastOpen = atom<{ isEnabled: boolean; source?: string }>({ isEnabled: false });

export const useHFForbiddenToast = () => {
  const { t } = useTranslation();
  const toast = useToast();
  const isHFForbiddenToastOpen = useStore($isHFForbiddenToastOpen);

  useEffect(() => {
    if (!isHFForbiddenToastOpen.isEnabled) {
      toast.close(TOAST_ID);
      return;
    }

    if (isHFForbiddenToastOpen.isEnabled) {
      toast({
        id: TOAST_ID,
        title: t('modelManager.hfForbidden'),
        description: (
          <Text fontSize="md">
            {t('modelManager.hfForbiddenErrorMessage')}
            <ExternalLink
              label={isHFForbiddenToastOpen.source || ''}
              href={`https://huggingface.co/${isHFForbiddenToastOpen.source}`}
            />
          </Text>
        ),
        status: 'error',
        isClosable: true,
        duration: null,
        onCloseComplete: () => $isHFForbiddenToastOpen.set({ isEnabled: false }),
      });
    }
  }, [isHFForbiddenToastOpen, t, toast]);
};
