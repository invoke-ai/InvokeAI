/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { AnyModelDefaultSettings } from '@features/models/core/types';

import { createListCollection, Grid, HStack, Icon, NumberInput, Stack, Switch, Text } from '@chakra-ui/react';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { useScopedAction } from '@features/models/ui/shared/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, Combobox, FieldLabel, Panel, Select } from '@platform/ui';
import { MoveHorizontalIcon } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { DefaultSettingsControl, DefaultSettingsModel } from './defaultSettingsFields';

import { getFieldsForModel, validateDefaults } from './defaultSettingsFields';

/**
 * Per-model generation defaults ("use these settings when this model is
 * selected"). Every field is individually toggleable: off = inherit the app
 * default (stored as null). Field policy and validation live in
 * `defaultSettingsFields.ts`; this file only renders and saves.
 */

interface DefaultSettingsDraft {
  modelKey: string;
  settings: AnyModelDefaultSettings;
  source: AnyModelDefaultSettings | null | undefined;
}

interface FieldControlProps {
  control: DefaultSettingsControl;
  disabled: boolean;
  label: string;
  setValue: (value: unknown) => void;
  value: unknown;
}

// Built per option list; memoized by the rendering component per field key.
const buildCollection = (options: readonly string[]) =>
  createListCollection({ items: options.map((option) => ({ label: option, value: option })) });

const FieldControl = ({ control, disabled, label, setValue, value }: FieldControlProps) => {
  const selectCollection = useMemo(
    () => (control.kind === 'select' ? buildCollection(control.options) : null),
    [control]
  );
  const comboboxOptions = useMemo(
    () => (control.kind === 'combobox' ? control.options.map((option) => ({ label: option, value: option })) : null),
    [control]
  );

  if (control.kind === 'number') {
    return (
      <NumberInput.Root
        disabled={disabled}
        max={control.max}
        min={control.min}
        position="relative"
        size="sm"
        step={control.step ?? 1}
        value={typeof value === 'number' ? String(value) : ''}
        w="full"
        onValueChange={(details) => {
          // Empty/partial input is transient; only commit finite numbers so the
          // field never silently flips back to "off".
          if (Number.isFinite(details.valueAsNumber)) {
            setValue(details.valueAsNumber);
          }
        }}
      >
        <NumberInput.Control />
        {/* Drag horizontally on the handle to scrub the value. */}
        <NumberInput.Scrubber
          alignItems="center"
          bottom="0"
          cursor="ew-resize"
          display="flex"
          left="2"
          position="absolute"
          top="0"
          zIndex={1}
        >
          <Icon as={MoveHorizontalIcon} boxSize="3" color="fg.subtle" />
        </NumberInput.Scrubber>
        <NumberInput.Input ps="7" />
      </NumberInput.Root>
    );
  }

  if (control.kind === 'select' && selectCollection) {
    return (
      <Select
        aria-label={label}
        collection={selectCollection}
        disabled={disabled}
        size="sm"
        value={typeof value === 'string' ? [value] : []}
        onValueChange={({ value: next }) => {
          const nextValue = next[0];

          if (nextValue !== undefined) {
            setValue(nextValue);
          }
        }}
      />
    );
  }

  if (control.kind === 'combobox' && comboboxOptions) {
    return (
      <Combobox
        aria-label={label}
        disabled={disabled}
        options={comboboxOptions}
        size="sm"
        value={typeof value === 'string' ? value : null}
        onValueChange={setValue}
      />
    );
  }

  return null;
};

export const DefaultSettingsSection = ({
  model,
  onError,
  onSaved,
}: {
  model: DefaultSettingsModel;
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  const fields = useMemo(() => getFieldsForModel(model), [model]);
  const [draft, setDraft] = useState<DefaultSettingsDraft>(() => ({
    modelKey: model.key,
    settings: { ...model.default_settings },
    source: model.default_settings,
  }));
  const [error, setError] = useState<string | null>(null);
  const { isBusy: isSaving, run } = useScopedAction();
  const isDraftCurrent = draft.modelKey === model.key && draft.source === model.default_settings;
  const savedSettingsDraft = useMemo(() => ({ ...model.default_settings }), [model.default_settings]);
  const settings = isDraftCurrent ? draft.settings : savedSettingsDraft;
  const visibleError = isDraftCurrent ? error : null;
  const visibleIsSaving = isDraftCurrent ? isSaving : false;

  const isDirty = useMemo(() => {
    const saved = model.default_settings ?? {};

    return fields.some((field) => (settings[field.key] ?? null) !== (saved[field.key] ?? null));
  }, [fields, model.default_settings, settings]);

  const setFieldValue = (key: keyof AnyModelDefaultSettings, value: unknown) => {
    setDraft({
      modelKey: model.key,
      settings: { ...settings, [key]: value as never },
      source: model.default_settings,
    });
    setError(null);
  };

  const handleSave = async () => {
    const validationError = validateDefaults(model, settings, t);

    if (validationError) {
      setError(validationError);
      return;
    }

    await run(
      async (owner) => {
        const updated = await updateModel(model.key, { default_settings: settings }, owner.signal);

        assertAccountScopeCurrent(owner);
        replaceModelInStore(updated);
        onSaved();
      },
      (_message, saveError) =>
        onError(saveError instanceof Error ? saveError.message : t('models.failedToSaveDefaults'))
    );
  };

  return (
    <Stack gap="3">
      <HStack justify="space-between">
        <Stack gap="0.5">
          <FieldLabel>{t('models.defaultSettings')}</FieldLabel>
          <Text color="fg.subtle" fontSize="2xs">
            {t('models.defaultSettingsHelp')}
          </Text>
        </Stack>
        <Button
          disabled={!isDirty}
          loading={visibleIsSaving}
          size="xs"
          variant="solid"
          onClick={() => void handleSave()}
        >
          {t('models.saveDefaults')}
        </Button>
      </HStack>
      {visibleError ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {visibleError}
        </Text>
      ) : null}
      <Grid gap="2.5" templateColumns="repeat(auto-fill, minmax(13rem, 1fr))">
        {fields.map((field) => {
          const value = settings[field.key] ?? null;
          const isEnabled = value !== null && value !== undefined;

          return (
            <Panel key={field.key} gap="2" p="2.5" tone="control">
              <HStack justify="space-between">
                <Text fontSize="2xs" fontWeight="600" textTransform="uppercase">
                  {t(field.labelKey)}
                </Text>
                <Switch.Root
                  checked={isEnabled}
                  colorPalette="accent"
                  size="xs"
                  onCheckedChange={(event) => setFieldValue(field.key, event.checked ? field.defaultValue : null)}
                >
                  <Switch.HiddenInput />
                  <Switch.Control>
                    <Switch.Thumb />
                  </Switch.Control>
                  <Switch.Label srOnly>{t('models.customizeDefaultField', { field: t(field.labelKey) })}</Switch.Label>
                </Switch.Root>
              </HStack>
              {field.control ? (
                <FieldControl
                  control={field.control}
                  disabled={!isEnabled}
                  label={t(field.labelKey)}
                  setValue={(nextValue) => setFieldValue(field.key, nextValue)}
                  value={isEnabled ? value : field.defaultValue}
                />
              ) : null}
              <Text color="fg.subtle" fontSize="2xs">
                {isEnabled ? t('models.customizedForThisModel') : t(field.inheritLabelKey)}
              </Text>
            </Panel>
          );
        })}
      </Grid>
    </Stack>
  );
};
