import type { ModelConfig } from '@features/models/core/types';

import { Stack, Switch, Text } from '@chakra-ui/react';
import { updateModel } from '@features/models/data/api';
import { patchModelInStore, replaceModelInStore } from '@features/models/data/modelsStore';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type VaeCpuOnlyModel = Pick<ModelConfig, 'cpu_only' | 'key' | 'name'>;

export const supportsVaeCpuOnlySetting = (model: Pick<ModelConfig, 'type'>): boolean => model.type === 'vae';

/** Immediately persists the VAE's CPU-only preference with optimistic rollback. */
export const VaeCpuOnlySetting = ({
  model,
  onError,
  onSaved,
}: {
  model: VaeCpuOnlyModel;
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  const [isPending, setIsPending] = useState(false);

  const handleCheckedChange = useCallback(
    async (details: { checked: boolean }) => {
      if (isPending) {
        return;
      }

      const owner = captureAccountScope();
      const previousCpuOnly = model.cpu_only;
      const cpuOnly = details.checked ? true : null;

      setIsPending(true);
      patchModelInStore(model.key, { cpu_only: cpuOnly });

      try {
        const updated = await updateModel(model.key, { cpu_only: cpuOnly }, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
        onSaved();
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        patchModelInStore(model.key, { cpu_only: previousCpuOnly });
        onError(getApiErrorMessage(error, t('models.failedToSaveVaeCpuSetting')));
      } finally {
        if (isAccountScopeCurrent(owner)) {
          setIsPending(false);
        }
      }
    },
    [isPending, model.cpu_only, model.key, onError, onSaved, t]
  );

  return (
    <Stack gap="1.5">
      <Switch.Root
        checked={model.cpu_only === true}
        colorPalette="accent"
        disabled={isPending}
        size="sm"
        onCheckedChange={handleCheckedChange}
      >
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label fontSize="xs" fontWeight="600">
          {t('models.runVaeOnCpu')}
        </Switch.Label>
      </Switch.Root>
      <Text color="fg.subtle" fontSize="2xs">
        {t('models.runVaeOnCpuHelp')}
      </Text>
    </Stack>
  );
};
