import type { GenerateModelConfig, GenerateSettings, VaePrecision } from '@workbench/generation/types';
import type { ChangeEvent } from 'react';

import { Badge, HStack, Input, InputGroup, NativeSelect, Switch } from '@chakra-ui/react';
import { Field } from '@workbench/components/ui';
import { getDefaultGenerateSettings, getGenerationUiPolicy } from '@workbench/generation/baseGenerationPolicies';
import { isVaeModelConfig } from '@workbench/generation/settings';
import { ModelSelect } from '@workbench/models/components';

import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';
import { ModelDefaultButton } from './shared/ModelDefaultButton';

interface GenerateAdvancedFieldsProps {
  settings: GenerateSettings;
  selectedModel: GenerateModelConfig | undefined;
  onCommit: (patch: Partial<GenerateSettings>) => void;
}

const SeamlessSwitch = ({
  checked,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
}) => (
  <Switch.Root checked={checked} flex="1" size="sm" onCheckedChange={(event) => onCheckedChange(event.checked)}>
    <Switch.HiddenInput />
    <Switch.Control _checked={{ bg: 'accent.solid' }}>
      <Switch.Thumb />
    </Switch.Control>
    <Switch.Label fontSize="xs">{label}</Switch.Label>
  </Switch.Root>
);

export const GenerateAdvancedFields = ({ onCommit, selectedModel, settings }: GenerateAdvancedFieldsProps) => {
  const modelBase = selectedModel?.base;
  const modelDefaults = selectedModel ? getDefaultGenerateSettings(selectedModel) : null;
  const policy = getGenerationUiPolicy(selectedModel, settings);
  const clipSkipMax = policy.clipSkipMax ?? 0;

  const updateNumber =
    (key: 'cfgRescaleMultiplier' | 'clipSkip', max: number) => (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.currentTarget.value);

      if (!Number.isFinite(value)) {
        return;
      }

      onCommit({ [key]: Math.min(max, Math.max(0, value)) });
    };

  const customVae = policy.sdVaeVisible && Boolean(settings.vae?.key);

  if (
    !policy.sdVaeVisible &&
    !policy.vaePrecisionVisible &&
    !policy.seamlessVisible &&
    !policy.clipSkipMax &&
    !policy.cfgRescaleVisible
  ) {
    return null;
  }

  const badges = (
    <>
      {settings.seamlessXAxis && <Badge size="xs">Tile X</Badge>}
      {settings.seamlessYAxis && <Badge size="xs">Tile Y</Badge>}
      {customVae && (
        <Badge maxW="32" size="xs" truncate>
          {settings.vae?.name}
        </Badge>
      )}
    </>
  );

  return (
    <GenerateCollapsibleSection label="Advanced" defaultOpen={false} badges={badges}>
      {policy.sdVaeVisible || policy.vaePrecisionVisible ? (
        <HStack alignItems="flex-start" gap="2" p="2">
          {policy.sdVaeVisible ? (
            <Field flex="2" label="VAE" helpText={settings.vae ? undefined : 'Using the VAE bundled with the model.'}>
              <HStack gap="1">
                <ModelSelect
                  filter={(model) => model.base === modelBase}
                  modelTypes={['vae']}
                  size="xs"
                  placeholder="Model default"
                  value={settings.vae?.key ?? null}
                  onChange={(model) => onCommit({ vae: isVaeModelConfig(model) ? model : null })}
                />
                <ModelDefaultButton
                  disabled={!settings.vae}
                  label="Use model default VAE"
                  onClick={() => onCommit({ vae: null })}
                />
              </HStack>
            </Field>
          ) : null}
          {policy.vaePrecisionVisible ? (
            <Field flex="1" label="VAE precision">
              <HStack gap="1">
                <NativeSelect.Root flex="1" size="xs">
                  <NativeSelect.Field
                    aria-label="VAE precision"
                    value={settings.vaePrecision}
                    onChange={(event: ChangeEvent<HTMLSelectElement>) =>
                      onCommit({ vaePrecision: event.currentTarget.value as VaePrecision })
                    }
                  >
                    <option value="fp16">FP16</option>
                    <option value="fp32">FP32</option>
                  </NativeSelect.Field>
                  <NativeSelect.Indicator />
                </NativeSelect.Root>
                <ModelDefaultButton
                  disabled={!modelDefaults || settings.vaePrecision === modelDefaults.vaePrecision}
                  label="Use model default VAE precision"
                  onClick={() => {
                    if (modelDefaults) {
                      onCommit({ vaePrecision: modelDefaults.vaePrecision });
                    }
                  }}
                />
              </HStack>
            </Field>
          ) : null}
        </HStack>
      ) : null}

      {policy.clipSkipMax || policy.cfgRescaleVisible ? (
        <HStack gap="2" p="2">
          {policy.clipSkipMax ? (
            <Field label="CLIP skip">
              <Input
                max={String(clipSkipMax)}
                min="0"
                size="xs"
                type="number"
                value={settings.clipSkip}
                onChange={updateNumber('clipSkip', clipSkipMax)}
              />
            </Field>
          ) : null}
          {policy.cfgRescaleVisible ? (
            <Field label="CFG rescale">
              <InputGroup
                endElement={
                  <ModelDefaultButton
                    disabled={!modelDefaults || settings.cfgRescaleMultiplier === modelDefaults.cfgRescaleMultiplier}
                    label="Use model default CFG rescale"
                    onClick={() => {
                      if (modelDefaults) {
                        onCommit({ cfgRescaleMultiplier: modelDefaults.cfgRescaleMultiplier });
                      }
                    }}
                  />
                }
                endElementProps={{ pointerEvents: 'auto' }}
              >
                <Input
                  max="0.99"
                  min="0"
                  size="xs"
                  step="0.05"
                  type="number"
                  value={settings.cfgRescaleMultiplier}
                  onChange={updateNumber('cfgRescaleMultiplier', 0.99)}
                />
              </InputGroup>
            </Field>
          ) : null}
        </HStack>
      ) : null}

      {policy.seamlessVisible ? (
        <Field label="Seamless tiling" p="2">
          <HStack gap="4">
            <SeamlessSwitch
              checked={settings.seamlessXAxis}
              label="X axis"
              onCheckedChange={(checked) => onCommit({ seamlessXAxis: checked })}
            />
            <SeamlessSwitch
              checked={settings.seamlessYAxis}
              label="Y axis"
              onCheckedChange={(checked) => onCommit({ seamlessYAxis: checked })}
            />
          </HStack>
        </Field>
      ) : null}
    </GenerateCollapsibleSection>
  );
};
