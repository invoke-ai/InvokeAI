import type { GenerationDevicesSetting } from '@features/queue/devices';

import { Spinner, Stack, Switch, Text } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import {
  getDeviceNameLabels,
  refreshGenerationDevices,
  updateGenerationDevices,
  useGenerationDevices,
} from '@features/queue/devices';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useCallback, useMemo, useState } from 'react';

/**
 * Which accelerators the server uses for generation (`generation_devices`).
 *
 * Server-wide app config rather than a user preference, so it is admin-only and only
 * takes effect after a restart — both of which the copy states plainly instead of
 * pretending the change is live.
 */
export const GenerationDevicesSettings = () => {
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

  const save = useCallback(async (next: GenerationDevicesSetting) => {
    setSaveError(null);
    try {
      await updateGenerationDevices(next);
      setDidChange(true);
    } catch (saveFailure) {
      setSaveError(saveFailure instanceof Error ? saveFailure.message : 'Could not save generation devices.');
    }
  }, []);

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
        setSaveError('Select at least one generation device.');
        return;
      }

      void save(next);
    },
    [options, save, selectedDevices]
  );

  if (loadState === 'loading' || loadState === 'idle') {
    return <Spinner size="sm" />;
  }

  if (error) {
    return (
      <Text color="fg.error" fontSize="xs">
        {error}
      </Text>
    );
  }

  if (options.length <= 1) {
    return (
      <Text color="fg.subtle" fontSize="xs">
        {options.length === 1
          ? `Generation runs on ${labels[options[0].device] ?? options[0].device}. Parallel generation needs more than one accelerator.`
          : 'No accelerators were detected, so generation runs on a single device.'}
      </Text>
    );
  }

  if (!canManageAppConfig) {
    return (
      <Stack gap="1">
        <Text color="fg" fontSize="xs">
          {isAuto
            ? 'Every available accelerator is used for generation.'
            : selectedDevices.map((device) => labels[device] ?? device).join(', ')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          Only an administrator can change which accelerators are used.
        </Text>
      </Stack>
    );
  }

  return (
    <Stack gap="3">
      <DeviceSwitch
        checked={isAuto}
        description="New accelerators are picked up automatically."
        label="Use every available accelerator"
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
        <Text color="fg.error" fontSize="xs">
          {saveError}
        </Text>
      ) : null}
      {didChange ? (
        <Text color="fg.subtle" fontSize="xs">
          Restart InvokeAI for changes to take effect.
        </Text>
      ) : null}
    </Stack>
  );
};

/**
 * One device's switch. Binds the device to the shared handler in its own scope so the
 * list does not create a fresh callback per row on every render.
 */
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
  onChange,
}: {
  checked: boolean;
  description?: string;
  label: string;
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
        <Switch.Label color="fg" fontSize="sm" fontWeight="500">
          {label}
        </Switch.Label>
        {description ? (
          <Text color="fg.subtle" fontSize="xs">
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
