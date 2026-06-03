import {
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  Tag,
  TagCloseButton,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useGetGenerationDeviceOptionsQuery,
  useGetRuntimeConfigQuery,
  useUpdateRuntimeConfigMutation,
} from 'services/api/endpoints/appInfo';

const AUTO = 'auto';

type GenerationDevicesValue = 'auto' | string[];

/** Drop the verbose vendor prefix so e.g. "NVIDIA GeForce RTX 3090" reads as "RTX 3090". */
const shortenDeviceName = (name: string): string => name.replace(/^NVIDIA GeForce /, '').replace(/^NVIDIA /, '');

type DeviceBadge = {
  /** The device identifier, or 'auto' for the special "use all GPUs" badge. */
  device: string;
  /** The label shown on the badge. */
  label: string;
  /** A human-readable description shown on hover (e.g. the GPU model name). */
  tooltip?: string;
};

export const SettingsGenerationDevices = memo(() => {
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: runtimeConfig } = useGetRuntimeConfigQuery();
  const { data: deviceOptions } = useGetGenerationDeviceOptionsQuery();
  const [updateRuntimeConfig, { isLoading }] = useUpdateRuntimeConfigMutation();

  const generationDevices: GenerationDevicesValue = runtimeConfig?.config.generation_devices ?? AUTO;
  const isAuto = generationDevices === AUTO;
  const selectedDevices = useMemo<string[]>(() => (isAuto ? [] : [...generationDevices]), [generationDevices, isAuto]);

  const canEditRuntimeConfig = runtimeConfig ? !runtimeConfig.config.multiuser || currentUser?.is_admin : false;
  const isDisabled = !runtimeConfig || !canEditRuntimeConfig || isLoading;

  const save = useCallback(
    async (value: GenerationDevicesValue) => {
      try {
        await updateRuntimeConfig({ generation_devices: value }).unwrap();
      } catch {
        toast({
          id: 'SETTINGS_GENERATION_DEVICES_SAVE_FAILED',
          title: t('settings.generationDevicesSaveFailed'),
          status: 'error',
        });
      }
    },
    [t, updateRuntimeConfig]
  );

  const autoBadge = useMemo<DeviceBadge>(() => ({ device: AUTO, label: t('settings.generationDevicesAuto') }), [t]);

  // Build a per-device badge (label + tooltip) keyed by device id, e.g. "cuda:0 (RTX 3090 #1)".
  // Cards sharing a name get a 1-based "#N" suffix so identical GPUs can be told apart.
  const deviceBadges = useMemo<Record<string, DeviceBadge>>(() => {
    const options = deviceOptions ?? [];
    const nameCounts = new Map<string, number>();
    for (const option of options) {
      const name = shortenDeviceName(option.name);
      nameCounts.set(name, (nameCounts.get(name) ?? 0) + 1);
    }
    const ordinals = new Map<string, number>();
    const badges: Record<string, DeviceBadge> = {};
    for (const option of options) {
      const name = shortenDeviceName(option.name);
      const ordinal = (ordinals.get(name) ?? 0) + 1;
      ordinals.set(name, ordinal);
      const namePart = (nameCounts.get(name) ?? 0) > 1 ? `${name} #${ordinal}` : name;
      badges[option.device] = { device: option.device, label: `${option.device} (${namePart})`, tooltip: option.name };
    }
    return badges;
  }, [deviceOptions]);

  // Fall back to a bare device id when a configured device isn't in the current options (e.g. a
  // GPU that's no longer present).
  const getDeviceBadge = useCallback(
    (device: string): DeviceBadge => deviceBadges[device] ?? { device, label: device },
    [deviceBadges]
  );

  // The active badges: the `auto` pseudo-device, or the explicitly-selected devices in config order.
  const activeBadges = useMemo<DeviceBadge[]>(() => {
    if (isAuto) {
      return [autoBadge];
    }
    return selectedDevices.map(getDeviceBadge);
  }, [autoBadge, getDeviceBadge, isAuto, selectedDevices]);

  // The inactive badges: `auto` (when an explicit list is active) plus any unselected devices.
  const inactiveBadges = useMemo<DeviceBadge[]>(() => {
    const badges: DeviceBadge[] = [];
    if (!isAuto) {
      badges.push(autoBadge);
    }
    for (const option of deviceOptions ?? []) {
      if (!selectedDevices.includes(option.device)) {
        badges.push(getDeviceBadge(option.device));
      }
    }
    return badges;
  }, [autoBadge, deviceOptions, getDeviceBadge, isAuto, selectedDevices]);

  const onActivate = useCallback(
    (device: string) => {
      if (isDisabled) {
        return;
      }
      if (device === AUTO) {
        save(AUTO);
        return;
      }
      // Switching from `auto` starts a fresh explicit list; otherwise append to the current selection.
      const next = isAuto ? [device] : Array.from(new Set([...selectedDevices, device]));
      save(next);
    },
    [isAuto, isDisabled, save, selectedDevices]
  );

  const onDeactivate = useCallback(
    (device: string) => {
      if (isDisabled) {
        return;
      }
      const next = selectedDevices.filter((d) => d !== device);
      // Never leave an empty selection — fall back to `auto`, which is always meaningful.
      save(next.length > 0 ? next : AUTO);
    },
    [isDisabled, save, selectedDevices]
  );

  return (
    <FormControl orientation="vertical">
      <FormLabel>{t('settings.generationDevices')}</FormLabel>
      <Flex w="full" gap={2} flexWrap="wrap" minH="32px" borderRadius="base" borderWidth={1} p={2} alignItems="center">
        {activeBadges.map((badge) => (
          <DeviceTag
            key={`active-${badge.device}`}
            badge={badge}
            isActive={true}
            // The `auto` badge is exclusive — it cannot be removed directly; pick a device to leave it.
            isClosable={badge.device !== AUTO}
            isDisabled={isDisabled}
            onActivate={onActivate}
            onDeactivate={onDeactivate}
          />
        ))}
      </Flex>
      {inactiveBadges.length > 0 && (
        <Flex gap={2} flexWrap="wrap">
          {inactiveBadges.map((badge) => (
            <DeviceTag
              key={`inactive-${badge.device}`}
              badge={badge}
              isActive={false}
              isClosable={false}
              isDisabled={isDisabled}
              onActivate={onActivate}
              onDeactivate={onDeactivate}
            />
          ))}
        </Flex>
      )}
      <FormHelperText>
        {t('settings.generationDevicesHelp')}{' '}
        <Text as="span" fontWeight="bold">
          {t('settings.generationDevicesRestart')}
        </Text>
      </FormHelperText>
    </FormControl>
  );
});

SettingsGenerationDevices.displayName = 'SettingsGenerationDevices';

type DeviceTagProps = {
  badge: DeviceBadge;
  isActive: boolean;
  isClosable: boolean;
  isDisabled: boolean;
  onActivate: (device: string) => void;
  onDeactivate: (device: string) => void;
};

const DeviceTag = memo(({ badge, isActive, isClosable, isDisabled, onActivate, onDeactivate }: DeviceTagProps) => {
  const onClick = useCallback(() => {
    if (isDisabled) {
      return;
    }
    if (isActive) {
      // An active, non-closable badge (the exclusive `auto`) is a no-op when clicked.
      if (isClosable) {
        onDeactivate(badge.device);
      }
    } else {
      onActivate(badge.device);
    }
  }, [badge.device, isActive, isClosable, isDisabled, onActivate, onDeactivate]);

  const isInteractive = !isDisabled && (!isActive || isClosable);

  const tag = (
    <Tag
      h="min-content"
      borderRadius="base"
      onClick={onClick}
      colorScheme={isActive ? 'invokeBlue' : 'base'}
      userSelect="none"
      role={isInteractive ? 'button' : undefined}
      cursor={isInteractive ? 'pointer' : 'default'}
      opacity={isDisabled ? 0.5 : 1}
      size="md"
      color="base.900"
      bg={isActive ? 'invokeBlue.300' : 'base.300'}
    >
      <Text fontSize="sm" fontWeight="semibold">
        {badge.label}
      </Text>
      {isActive && isClosable && <TagCloseButton />}
    </Tag>
  );

  if (!badge.tooltip) {
    return tag;
  }

  return (
    <Tooltip label={badge.tooltip} placement="top">
      {tag}
    </Tooltip>
  );
});

DeviceTag.displayName = 'DeviceTag';
