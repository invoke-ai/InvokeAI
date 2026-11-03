import { useInstallOutcomes } from '@features/models/data/installsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

/** ModelInstallRuntime announces settled installs; notifications also work outside workbench providers. */
export const useInstallOutcomeToasts = (): void => {
  const { t } = useTranslation();
  const notify = useNotify();
  const outcomes = useInstallOutcomes();
  const seenOutcomeIdsRef = useRef<Set<number> | null>(null);

  useEffect(() => {
    if (seenOutcomeIdsRef.current === null) {
      seenOutcomeIdsRef.current = new Set(outcomes.map((outcome) => outcome.id));

      return;
    }

    for (const outcome of [...outcomes].reverse()) {
      if (seenOutcomeIdsRef.current.has(outcome.id)) {
        continue;
      }

      seenOutcomeIdsRef.current.add(outcome.id);

      if (outcome.kind === 'completed') {
        notify.success(t('models.modelInstalled'), outcome.modelName ?? outcome.source);
      } else if (outcome.kind === 'error') {
        notify.error(t('models.modelInstallFailed'), `${outcome.source}: ${outcome.error ?? t('common.unknownError')}`);
      } else {
        notify.info(t('models.modelInstallCancelled'), outcome.source);
      }
    }
  }, [notify, outcomes, t]);
};
