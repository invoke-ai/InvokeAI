import type { ModelConfig } from '@features/models/core/types';

import { Stack, Switch, Text } from '@chakra-ui/react';
import { updateModel } from '@features/models/data/api';
import { patchModelInStore, replaceModelInStore } from '@features/models/data/modelsStore';
import { useScopedAction } from '@features/models/ui/shared/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export type CpuOnlyModel = Pick<ModelConfig, 'cpu_only' | 'key' | 'name' | 'type'>;

/**
 * Every type whose backend config class carries a top-level `cpu_only` field.
 * The PATCH silently drops the field elsewhere (notably qwen_vl_encoder and
 * pid_decoder), so the toggle is only offered where it acts.
 */
const CPU_ONLY_TYPES: ReadonlySet<string> = new Set([
  'vae',
  'clip_embed',
  't5_encoder',
  'qwen3_encoder',
  'clip_vision',
  'siglip',
  'llava_onevision',
  'text_llm',
  'wan_t5_encoder',
  'gemma2_encoder',
  'mistral_encoder',
  'qwen3_vl_encoder',
]);

export const supportsCpuOnlySetting = (model: Pick<ModelConfig, 'type'>): boolean => CPU_ONLY_TYPES.has(model.type);

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
        size="sm"
        onCheckedChange={handleCheckedChange}
      >
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label fontSize="xs" fontWeight="600">
          {isVae ? t('models.runVaeOnCpu') : t('models.runOnCpu')}
        </Switch.Label>
      </Switch.Root>
      <Text color="fg.subtle" fontSize="2xs">
        {isVae ? t('models.runVaeOnCpuHelp') : t('models.runOnCpuHelp')}
      </Text>
    </Stack>
  );
};
