import type { GenerateSettings, MainModelConfig, VaePrecision } from '@workbench/generation/types';
import type { ChangeEvent } from 'react';

import { Badge, HStack, Input, NativeSelect, Switch, Text } from '@chakra-ui/react';
import { ModelSelect } from '@workbench/components/ModelSelect';
import { Field } from '@workbench/components/ui/Field';
import { CLIP_SKIP_MAX, isVaeModelConfig } from '@workbench/generation/settings';

import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';

interface GenerateAdvancedFieldsProps {
  settings: GenerateSettings;
  selectedModel: MainModelConfig | undefined;
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
  // CLIP skip and CFG rescale target the SD1/SD2 text encoder; SDXL conditioning does not use them.
  const supportsClipSkip = modelBase === 'sd-1' || modelBase === 'sd-2';
  const clipSkipMax = supportsClipSkip ? CLIP_SKIP_MAX[modelBase] : 0;

  const updateNumber =
    (key: 'cfgRescaleMultiplier' | 'clipSkip', max: number) => (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.currentTarget.value);

      if (!Number.isFinite(value)) {
        return;
      }

      onCommit({ [key]: Math.min(max, Math.max(0, value)) });
    };

  const customVae = Boolean(settings.vae?.key);

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
      <HStack alignItems="flex-start" gap="2" p="2">
        <Field flex="2" label="VAE" helpText={settings.vae ? undefined : 'Using the VAE bundled with the model.'}>
          <ModelSelect
            filter={(model) => model.base === modelBase}
            modelTypes={['vae']}
            size="xs"
            placeholder="Model default"
            value={settings.vae?.key ?? null}
            onChange={(model) => onCommit({ vae: isVaeModelConfig(model) ? model : null })}
          />
        </Field>
        <Field flex="1" label="VAE precision">
          <NativeSelect.Root size="xs">
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
        </Field>
      </HStack>

      {supportsClipSkip ? (
        <HStack gap="2" p="2">
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
          <Field label="CFG rescale">
            <Input
              max="0.99"
              min="0"
              size="xs"
              step="0.05"
              type="number"
              value={settings.cfgRescaleMultiplier}
              onChange={updateNumber('cfgRescaleMultiplier', 0.99)}
            />
          </Field>
        </HStack>
      ) : null}

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
      <Text color="fg.subtle" fontSize="2xs" p="2">
        LoRAs, reference images, and the SDXL refiner are not wired into the Generate graph yet.
      </Text>
    </GenerateCollapsibleSection>
  );
};
