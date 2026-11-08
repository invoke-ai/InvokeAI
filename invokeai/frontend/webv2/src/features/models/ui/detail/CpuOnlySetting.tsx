import type { ModelConfig } from '@features/models/core/types';

import { Stack, Switch, Text } from '@chakra-ui/react';
import { updateModel } from '@features/models/data/api';
import { patchModelInStore, replaceModelInStore } from '@features/models/data/modelsStore';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useCallback, useId } from 'react';
import { useTranslation } from 'react-i18next';

export type CpuOnlyModel = Pick<ModelConfig, 'cpu_only' | 'key' | 'name' | 'type'>;

/**
 * The backend serializes `cpu_only` (null when unset) on exactly the configs that store it, so its presence decides.
 * One model type can span configs with and without the field, so a type list would offer a switch the PATCH rejects.
 */
export const supportsCpuOnlySetting = (model: Pick<ModelConfig, 'cpu_only'>): boolean => model.cpu_only !== undefined;

/** Immediately persists the model's CPU-only preference with optimistic rollback. */
export const CpuOnlySetting = ({
  model,
  onError,
  onSaved,
}: {
  model: CpuOnlyModel;
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  const { isBusy: isPending, run } = useScopedAction();
  const isVae = model.type === 'vae';
  const helpId = useId();

  const handleCheckedChange = useCallback(
    async (details: { checked: boolean }) => {
      if (isPending) {
        return;
      }

      const previousCpuOnly = model.cpu_only;
      const cpuOnly = details.checked ? true : null;

      patchModelInStore(model.key, { cpu_only: cpuOnly });

      await run(
        async (owner) => {
          const updated = await updateModel(model.key, { cpu_only: cpuOnly }, owner.signal);

          assertAccountScopeCurrent(owner);
          replaceModelInStore(updated);
          onSaved();
        },
        (_message, error) => {
          patchModelInStore(model.key, { cpu_only: previousCpuOnly });
          onError(getApiErrorMessage(error, t('models.failedToSaveCpuSetting')));
        }
      );
    },
    [isPending, model.cpu_only, model.key, onError, onSaved, run, t]
  );

  return (
    <Stack gap="1.5">
      <Switch.Root
        checked={model.cpu_only === true}
        colorPalette="accent"
        disabled={isPending}
        onCheckedChange={handleCheckedChange}
      >
        <Switch.HiddenInput aria-describedby={helpId} />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label fontSize="md" fontWeight="600">
          {isVae ? t('models.runVaeOnCpu') : t('models.runOnCpu')}
        </Switch.Label>
      </Switch.Root>
      <Text color="fg.subtle" fontSize="xs" id={helpId}>
        {isVae ? t('models.runVaeOnCpuHelp') : t('models.runOnCpuHelp')}
      </Text>
    </Stack>
  );
};
