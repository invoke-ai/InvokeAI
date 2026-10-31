import type { GenerationDevicesSetting } from '@features/queue/devices';

import { HStack, Spinner, Stack, Switch, Text } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import {
  getDeviceNameLabels,
  refreshGenerationDevices,
  updateGenerationDevices,
  useGenerationDevices,
} from '@features/queue/devices';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ModifiedSettingIndicator } from '@platform/ui/settings/ModifiedSettingIndicator';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Generation devices are admin-only server configuration and take effect after restart. */
export const GenerationDevicesSettings = () => {
  const { t } = useTranslation();
  const { canManageAppConfig } = useCapabilities();
  const { error, loadState, options, setting } = useGenerationDevices();
  const [saveError, setSaveError] = useState<string | null>(null);
  const [didChange, setDidChange] = useState(false);

  useMountEffect(() => {
    void refreshGenerationDevices();
  });

  const labels = useMemo(() => getDeviceNameLabels(options), [options]);
  const isAuto = setting === 'auto' || setting === null;
  const selectedDevices = useMemo(
    () => (Array.isArray(setting) ? setting : options.map((option) => option.device)),
    [options, setting]
  );

  const save = useCallback(
    async (next: GenerationDevicesSetting) => {
      setSaveError(null);
      try {
        await updateGenerationDevices(next);
        setDidChange(true);
      } catch (saveFailure) {
        setSaveError(saveFailure instanceof Error ? saveFailure.message : t('settings.generationDevices.couldNotSave'));
      }
    },
    [t]
  );

  const toggleAuto = useCallback(
    (checked: boolean) => {
      // Leaving auto starts from the devices in effect right now, so turning the
      // switch off is not itself a change in which accelerators are used.
      void save(checked ? 'auto' : selectedDevices);
    },
    [save, selectedDevices]
  );

  const toggleDevice = useCallback(
    (device: string, checked: boolean) => {
      const next = checked
        ? options
            .map((option) => option.device)
            .filter((candidate) => selectedDevices.includes(candidate) || candidate === device)
        : selectedDevices.filter((candidate) => candidate !== device);

      // The backend rejects an empty list, and a config with no devices would fail on
      // the next startup — so refuse locally rather than surfacing a 422.
      if (next.length === 0) {
        setSaveError(t('settings.generationDevices.selectAtLeastOne'));
        return;
      }

      void save(next);
    },
    [options, save, selectedDevices, t]
  );

  if (loadState === 'loading' || loadState === 'idle') {
    return <Spinner size="lg" />;
  }

  if (error) {
    return (
      <Text color="fg.error" fontSize="md">
        {error}
      </Text>
    );
  }

  if (options.length <= 1) {
    return (
      <HStack align="start" gap="2">
        <Text color="fg.muted" fontSize="md">
          {options.length === 1
            ? t('settings.generationDevices.singleDevice', { device: labels[options[0].device] ?? options[0].device })
            : t('settings.generationDevices.noAccelerators')}
        </Text>
        {!isAuto ? <ModifiedSettingIndicator label={t('settings.catalog.generationDevices.label')} /> : null}
      </HStack>
    );
  }

  if (!canManageAppConfig) {
    return (
      <Stack gap="1">
        <HStack gap="2">
          <Text color="fg" fontSize="md">
            {isAuto
              ? t('settings.generationDevices.usingAll')
              : selectedDevices.map((device) => labels[device] ?? device).join(', ')}
          </Text>
          {!isAuto ? <ModifiedSettingIndicator label={t('settings.catalog.generationDevices.label')} /> : null}
        </HStack>
        <Text color="fg.muted" fontSize="md">
          {t('settings.generationDevices.adminOnly')}
        </Text>
      </Stack>
    );
  }

  return (
    <Stack gap="3">
      <DeviceSwitch
        checked={isAuto}
        description={t('settings.generationDevices.useAllDescription')}
        label={t('settings.generationDevices.useAll')}
        isModified={!isAuto}
        onChange={toggleAuto}
      />
      {isAuto ? null : (
        <Stack gap="2" ps="4">
          {options.map((option) => (
            <DeviceToggle
              key={option.device}
              checked={selectedDevices.includes(option.device)}
              device={option.device}
              label={labels[option.device] ?? option.device}
              onToggle={toggleDevice}
            />
          ))}
        </Stack>
      )}
      {saveError ? (
        <Text color="fg.error" fontSize="md">
          {saveError}
        </Text>
      ) : null}
      {didChange ? (
        <Text color="fg.muted" fontSize="md">
          {t('settings.generationDevices.restartRequired')}
        </Text>
      ) : null}
    </Stack>
  );
};

const DeviceToggle = ({
  checked,
  device,
  label,
  onToggle,
}: {
  checked: boolean;
  device: string;
  label: string;
  onToggle: (device: string, checked: boolean) => void;
}) => {
  const handleChange = useCallback((next: boolean) => onToggle(device, next), [device, onToggle]);

  return <DeviceSwitch checked={checked} label={label} onChange={handleChange} />;
};

/**
 * Each switch owns its own Switch.Root rather than sharing one Field.Root: sibling
 * switches under a single Field.Root share a generated id, and a label click then
 * toggles the wrong control.
 */
const DeviceSwitch = ({
  checked,
  description,
  label,
  isModified,
  onChange,
}: {
  checked: boolean;
  description?: string;
  label: string;
  isModified?: boolean;
  onChange: (checked: boolean) => void;
}) => {
  const handleCheckedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) => onChange(event.checked === true),
    [onChange]
  );

  return (
    <Switch.Root
      alignItems="center"
      checked={checked}
      display="flex"
      gap="4"
      justifyContent="space-between"
      w="full"
      onCheckedChange={handleCheckedChange}
    >
      <Stack gap="0.5">
        <HStack gap="2">
          <Switch.Label color="fg" fontSize="lg" fontWeight="500">
            {label}
          </Switch.Label>
          {isModified ? <ModifiedSettingIndicator label={label} /> : null}
        </HStack>
        {description ? (
          <Text color="fg.muted" fontSize="md">
            {description}
          </Text>
        ) : null}
      </Stack>
      <Switch.HiddenInput />
      <Switch.Control>
        <Switch.Thumb />
      </Switch.Control>
    </Switch.Root>
  );
};
