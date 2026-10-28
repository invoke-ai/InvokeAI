import { HStack, Icon, Spinner, Text } from '@chakra-ui/react';
import { refreshModels, useModelsSelector } from '@features/models/data/modelsStore';
import { useScopedAction } from '@platform/react/useScopedAction';
import { Button } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { FolderSearchIcon } from 'lucide-react';
import { useCallback, useId } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * One-click scan of the server's configured models folder. The path is the one the library already loads to resolve
 * relative model paths; the library load fetches it best-effort, so a missing path means that request failed and
 * reloading the library asks for it again.
 */
export const ModelsFolderScan = ({
  isRunning,
  isScanning,
  onScan,
  onStop,
}: {
  /** This row started the scan in flight, so it owns Stop. */
  isRunning: boolean;
  /** Any scan is in flight; one runs at a time. */
  isScanning: boolean;
  onScan: (path: string) => void;
  onStop: () => void;
}) => {
  const { t } = useTranslation();
  const modelsDir = useModelsSelector((snapshot) => snapshot.modelsDir);
  const libraryStatus = useModelsSelector((snapshot) => snapshot.status);
  const { isBusy: isRetrying, run: runRetry } = useScopedAction();
  const pathId = useId();
  const isLocating = modelsDir === null && (libraryStatus === 'idle' || libraryStatus === 'loading' || isRetrying);

  const scan = useCallback(() => {
    if (modelsDir) {
      onScan(modelsDir);
    }
  }, [modelsDir, onScan]);
  const retry = useCallback(() => void runRetry((owner) => refreshModels(owner)), [runRetry]);

  if (modelsDir === null && !isLocating) {
    return (
      <HStack gap="2" px="3">
        <Text color="fg.error" flex="1" fontSize="xs" minW="0" role="alert">
          {t('models.modelsFolderUnavailable')}
        </Text>
        <Button size="sm" variant="outline" onClick={retry}>
          {t('common.retry')}
        </Button>
      </HStack>
    );
  }

  return (
    <HStack gap="2" px="3">
      {isRunning ? (
        <Button aria-describedby={pathId} size="sm" variant="outline" onClick={onStop}>
          <Spinner size="inherit" />
          {t('models.stopScan')}
        </Button>
      ) : (
        <Button
          aria-describedby={pathId}
          disabled={modelsDir === null || isScanning}
          size="sm"
          variant="outline"
          onClick={scan}
        >
          <Icon as={FolderSearchIcon} boxSize="3.5" />
          {t('models.scanModelsFolder')}
        </Button>
      )}
      {modelsDir === null ? (
        <Text color="fg.subtle" fontSize="xs" id={pathId}>
          {t('common.loading')}
        </Text>
      ) : (
        <MiddleTruncate
          color="fg.muted"
          flex="1"
          fontFamily="mono"
          fontSize="xs"
          id={pathId}
          tailGraphemes={24}
          text={modelsDir}
        />
      )}
    </HStack>
  );
};
