import type { ThemeAdjustment } from '@theme/adjustments';

import { Collapsible, HStack, Slider, Stack, Text } from '@chakra-ui/react';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { ColorPicker } from '@platform/ui/ColorPicker';
import { themeSeedToPickerHex } from '@theme/adjustments';
import { THEMES_BY_ID } from '@theme/themes';
import { RotateCcwIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
  useWorkbenchSettingsSelector,
} from './store';

type LightnessKey =
  | 'backgroundLightness'
  | 'panelLightness'
  | 'inputLightness'
  | 'inputBorderLightness'
  | 'textLightness';
type ColorKey = 'brandHex' | 'accentHex';

const EMPTY_ADJUSTMENT: ThemeAdjustment = {};

/** The host's preference store is already account-owned and synchronized with the server. */
export const ThemeAdjustmentSettings = () => {
  const { t } = useTranslation();
  const [accountScope] = useState(captureAccountScope);
  const { themeAdjustments, themeId } = useWorkbenchPreferenceSelector(
    (preferences) => ({
      themeAdjustments: preferences.themeAdjustments,
      themeId: preferences.themeId,
    }),
    (previous, next) => previous.themeId === next.themeId && previous.themeAdjustments === next.themeAdjustments
  );
  const scope = useWorkbenchSettingsSelector((snapshot) => snapshot.scope);
  const adjustment = themeAdjustments[themeId] ?? EMPTY_ADJUSTMENT;
  const theme = THEMES_BY_ID[themeId];
  const hasChanges = Object.keys(adjustment).length > 0;

  const update = useCallback(
    (key: LightnessKey | ColorKey, value: number | string) => {
      if (!isAccountScopeCurrent(accountScope)) {
        return;
      }
      const current = getWorkbenchPreferences().themeAdjustments;
      void patchWorkbenchPreferences({
        themeAdjustments: { ...current, [themeId]: { ...current[themeId], [key]: value } },
      });
    },
    [accountScope, themeId]
  );
  const reset = useCallback(() => {
    if (!isAccountScopeCurrent(accountScope)) {
      return;
    }
    const next = { ...getWorkbenchPreferences().themeAdjustments };
    delete next[themeId];
    void patchWorkbenchPreferences({ themeAdjustments: next });
  }, [accountScope, themeId]);

  return (
    <Collapsible.Root defaultOpen={false} borderWidth="1px" borderColor="border" rounded="md">
      <Collapsible.Trigger px="3" py="2" w="full" textAlign="start" fontWeight="semibold">
        {t('settings.catalog.themeAdjustments.open', { defaultValue: 'Customize color and contrast' })}
      </Collapsible.Trigger>
      <Collapsible.Content>
        <Stack gap="3" p="3" borderTopWidth="1px" borderColor="border.subtle">
          <HStack justify="space-between">
            <Text color="fg.muted" fontSize="sm">
              {scope === 'user'
                ? t('settings.catalog.themeAdjustments.account', {
                    defaultValue: 'Saved for your account and selected theme.',
                  })
                : t('settings.catalog.themeAdjustments.install', {
                    defaultValue: 'Saved for this install and selected theme.',
                  })}
            </Text>
            <Button size="sm" variant="outline" disabled={!hasChanges} onClick={reset}>
              <RotateCcwIcon />
              {t('settings.catalog.themeAdjustments.reset', { defaultValue: 'Reset theme colors' })}
            </Button>
          </HStack>
          <LightnessSetting
            field="backgroundLightness"
            key={`${themeId}:backgroundLightness:${adjustment.backgroundLightness ?? 'default'}`}
            label={t('settings.catalog.themeAdjustments.background', { defaultValue: 'Application background' })}
            value={
              adjustment.backgroundLightness ??
              themeDefaultLightness(theme.colors.neutral[theme.colorScheme === 'light' ? 50 : 900])
            }
            onCommit={update}
          />
          <LightnessSetting
            field="panelLightness"
            key={`${themeId}:panelLightness:${adjustment.panelLightness ?? 'default'}`}
            label={t('settings.catalog.themeAdjustments.panels', { defaultValue: 'Section panels and menus' })}
            value={adjustment.panelLightness ?? themeDefaultLightness(theme.colors.control)}
            onCommit={update}
          />
          <LightnessSetting
            field="inputLightness"
            key={`${themeId}:inputLightness:${adjustment.inputLightness ?? 'default'}`}
            label={t('settings.catalog.themeAdjustments.inputs', { defaultValue: 'Input field brightness' })}
            value={adjustment.inputLightness ?? themeDefaultLightness(theme.colors.control)}
            onCommit={update}
          />
          <LightnessSetting
            field="inputBorderLightness"
            key={`${themeId}:inputBorderLightness:${adjustment.inputBorderLightness ?? 'default'}`}
            label={t('settings.catalog.themeAdjustments.inputBorders', { defaultValue: 'Input field borders' })}
            value={
              adjustment.inputBorderLightness ??
              themeDefaultLightness(theme.colors.neutral[theme.colorScheme === 'light' ? 300 : 600])
            }
            onCommit={update}
          />
          <LightnessSetting
            field="textLightness"
            key={`${themeId}:textLightness:${adjustment.textLightness ?? 'default'}`}
            label={t('settings.catalog.themeAdjustments.text', { defaultValue: 'Text brightness (black to white)' })}
            value={
              adjustment.textLightness ??
              themeDefaultLightness(theme.colors.neutral[theme.colorScheme === 'light' ? 950 : 50])
            }
            onCommit={update}
          />
          <ColorSetting
            field="brandHex"
            key={`${themeId}:brandHex:${adjustment.brandHex ?? 'default'}`}
            label={t('settings.catalog.themeAdjustments.brand', { defaultValue: 'Primary accent (Invoke)' })}
            value={adjustment.brandHex ?? themeSeedToPickerHex(theme.colors.brand.solid)}
            onCommit={update}
          />
          <ColorSetting
            field="accentHex"
            key={`${themeId}:accentHex:${adjustment.accentHex ?? 'default'}`}
            label={t('settings.catalog.themeAdjustments.accent', { defaultValue: 'Selection and focus color' })}
            value={adjustment.accentHex ?? themeSeedToPickerHex(theme.colors.accent.solid)}
            onCommit={update}
          />
          <Text color="fg.muted" fontSize="xs">
            {t('settings.catalog.themeAdjustments.highContrast', {
              defaultValue: 'High contrast temporarily uses the native palette; your adjustments remain saved.',
            })}
          </Text>
        </Stack>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};

const LightnessSetting = ({
  field,
  label,
  onCommit,
  value,
}: {
  field: LightnessKey;
  label: string;
  onCommit: (field: LightnessKey | ColorKey, value: number | string) => void;
  value: number;
}) => {
  const [draft, setDraft] = useState(value);
  const sliderLabel = useMemo(() => [label], [label]);
  const sliderValue = useMemo(() => [draft], [draft]);
  const change = useCallback((details: { value: number[] }) => setDraft(details.value[0] ?? value), [value]);
  const commit = useCallback(
    (details: { value: number[] }) => onCommit(field, details.value[0] ?? value),
    [field, onCommit, value]
  );
  return (
    <HStack gap="3" w="full">
      <Text as="span" flex="0 0 11rem" fontSize="sm">
        {label}
      </Text>
      <Slider.Root
        aria-label={sliderLabel}
        flex="1"
        min={0}
        max={100}
        step={1}
        value={sliderValue}
        onValueChange={change}
        onValueChangeEnd={commit}
      >
        <Slider.Control>
          <Slider.Track>
            <Slider.Range />
          </Slider.Track>
          <Slider.Thumbs />
        </Slider.Control>
      </Slider.Root>
      <Text fontSize="sm" minW="8" textAlign="end" fontVariantNumeric="tabular-nums">
        {draft}%
      </Text>
    </HStack>
  );
};

const ColorSetting = ({
  field,
  label,
  onCommit,
  value,
}: {
  field: ColorKey;
  label: string;
  onCommit: (field: LightnessKey | ColorKey, value: number | string) => void;
  value: string;
}) => {
  const [draft, setDraft] = useState(value);
  const commit = useCallback((next: string) => onCommit(field, next), [field, onCommit]);
  return (
    <HStack gap="3">
      <Text as="span" flex="0 0 11rem" fontSize="sm">
        {label}
      </Text>
      <ColorPicker aria-label={label} value={draft} withValueText onValueChange={setDraft} onValueChangeEnd={commit} />
    </HStack>
  );
};

const themeDefaultLightness = (color: string): number =>
  Math.round(Number(color.match(/^oklch\(([\d.]+)%/)?.[1] ?? 50));
