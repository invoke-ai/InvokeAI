/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig, PredictionType } from '@features/models/core/types';

import { HStack, Input, NativeSelect, Stack, Text, Textarea } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models/core/baseIdentity';
import { modelEditSchema, type ModelEditFormValues } from '@features/models/core/schemas';
import { getModelTypeLabel, MODEL_CATEGORIES } from '@features/models/core/taxonomy';
import { updateModel } from '@features/models/data/api';
import { replaceModelInStore } from '@features/models/data/modelsStore';
import { useZodForm } from '@platform/react/useZodForm';
import { Button, Field } from '@platform/ui';
import { useTranslation } from 'react-i18next';

const KNOWN_BASES = [
  'sd-1',
  'sd-2',
  'sd-3',
  'sdxl',
  'sdxl-refiner',
  'flux',
  'flux2',
  'cogview4',
  'qwen-image',
  'z-image',
  'anima',
  'any',
  'unknown',
];

/** Zod-validated editor for a model's identity fields. */
type ModelEditTarget = Pick<
  ModelConfig,
  'base' | 'description' | 'key' | 'name' | 'prediction_type' | 'source_url' | 'type' | 'variant'
>;

export const ModelEditForm = ({
  model,
  onCancel,
  onSaved,
}: {
  model: ModelEditTarget;
  onCancel: () => void;
  onSaved: () => void;
}) => {
  const { t } = useTranslation();
  const form = useZodForm(modelEditSchema, {
    base: String(model.base),
    description: model.description ?? '',
    name: model.name,
    predictionType: (model.prediction_type ?? '') as ModelEditFormValues['predictionType'],
    sourceUrl: model.source_url ?? '',
    type: String(model.type),
    variant: model.variant ?? '',
  });

  const bases = KNOWN_BASES.includes(String(model.base)) ? KNOWN_BASES : [String(model.base), ...KNOWN_BASES];

  const handleSave = () =>
    form.handleSubmit(async (values) => {
      const updated = await updateModel(model.key, {
        base: values.base,
        description: values.description || null,
        name: values.name,
        prediction_type: values.predictionType === '' ? null : (values.predictionType as PredictionType),
        source_url: values.sourceUrl === '' ? null : values.sourceUrl,
        type: values.type,
        variant: values.variant === '' ? null : values.variant,
      });

      replaceModelInStore(updated);
      onSaved();
    });

  return (
    <Stack gap="3">
      <Field error={form.errors.name} label={t('common.name')}>
        <Input
          aria-invalid={form.errors.name ? true : undefined}
          size="sm"
          value={form.values.name}
          onChange={(event) => form.setValue('name', event.currentTarget.value)}
        />
      </Field>
      <Field error={form.errors.description} label={t('models.description')}>
        <Textarea
          rows={2}
          size="sm"
          value={form.values.description}
          onChange={(event) => form.setValue('description', event.currentTarget.value)}
        />
      </Field>
      <HStack align="start" gap="2">
        <Field error={form.errors.base} label={t('models.base')}>
          <NativeSelect.Root size="sm">
            <NativeSelect.Field
              value={form.values.base}
              onChange={(event) => form.setValue('base', event.currentTarget.value)}
            >
              {bases.map((base) => (
                <option key={base} value={base}>
                  {getModelBaseLabel(base)}
                </option>
              ))}
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
        </Field>
        <Field error={form.errors.type} label={t('models.type')}>
          <NativeSelect.Root size="sm">
            <NativeSelect.Field
              value={form.values.type}
              onChange={(event) => form.setValue('type', event.currentTarget.value)}
            >
              {MODEL_CATEGORIES.map((category) => (
                <option key={category.type} value={category.type}>
                  {getModelTypeLabel(category.type)}
                </option>
              ))}
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
        </Field>
      </HStack>
      <HStack align="start" gap="2">
        <Field error={form.errors.variant} helpText={t('models.variantHelp')} label={t('models.variant')}>
          <Input
            size="sm"
            value={form.values.variant}
            onChange={(event) => form.setValue('variant', event.currentTarget.value)}
          />
        </Field>
        <Field error={form.errors.predictionType} label={t('models.predictionType')}>
          <NativeSelect.Root size="sm">
            <NativeSelect.Field
              value={form.values.predictionType}
              onChange={(event) =>
                form.setValue('predictionType', event.currentTarget.value as ModelEditFormValues['predictionType'])
              }
            >
              <option value="">{t('common.none')}</option>
              <option value="epsilon">epsilon</option>
              <option value="v_prediction">v_prediction</option>
              <option value="sample">sample</option>
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
        </Field>
      </HStack>
      <Field error={form.errors.sourceUrl} helpText={t('models.sourceUrlHelp')} label={t('models.sourceUrl')}>
        <Input
          aria-invalid={form.errors.sourceUrl ? true : undefined}
          placeholder="https://…"
          size="sm"
          value={form.values.sourceUrl}
          onChange={(event) => form.setValue('sourceUrl', event.currentTarget.value)}
        />
      </Field>
      {form.formError ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {form.formError}
        </Text>
      ) : null}
      <HStack gap="2" justify="flex-end">
        <Button disabled={form.isSubmitting} size="xs" variant="ghost" onClick={onCancel}>
          {t('common.cancel')}
        </Button>
        <Button loading={form.isSubmitting} size="xs" variant="solid" onClick={() => void handleSave()}>
          {t('users.saveChanges')}
        </Button>
      </HStack>
    </Stack>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
