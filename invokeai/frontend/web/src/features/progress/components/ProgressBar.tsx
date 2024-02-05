import { Progress } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useIsProcessing } from 'features/queue/hooks/useIsProcessing';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ProgressBar = () => {
  const { t } = useTranslation();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const isProcessing = useIsProcessing();
  const hasSteps = useAppSelector((s) => isProcessing && Boolean(s.progress.currentDenoiseProgress));
  const value = useAppSelector((s) => (isProcessing ? (s.progress.currentDenoiseProgress?.percentage ?? 0) * 100 : 0));
  const isIndeterminate = useMemo(() => {
    return isConnected && isProcessing && !hasSteps;
  }, [hasSteps, isConnected, isProcessing]);

  return (
    <Progress
      value={value}
      aria-label={t('accessibility.invokeProgressBar')}
      isIndeterminate={isIndeterminate}
      h={2}
      w="full"
      colorScheme="invokeBlue"
    />
  );
};

export default memo(ProgressBar);
