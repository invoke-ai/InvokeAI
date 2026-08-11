/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig } from '@features/models/core/types';
import type { ReactNode } from 'react';

import { chakra, DataList, HStack, Icon, Menu, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { isConvertibleToDiffusers } from '@features/models/core/baseIdentity';
import { isLinkableType } from '@features/models/core/relationships';
import { isAbsoluteModelPath, resolveModelAbsolutePath } from '@features/models/core/schemas';
import { formatBytes, getModelSourceHref } from '@features/models/core/taxonomy';
import { useModelsSelector, type ModelsSnapshot } from '@features/models/data/modelsStore';
import {
  ModelActionConfirmDialog,
  ModelActionMenuItems,
  type PendingModelAction,
} from '@features/models/ui/shared/ModelActionsMenu';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { areArraysEqual } from '@platform/state/selectors';
import { Button, IconButton, MenuContent } from '@platform/ui';
import { ExternalLinkIcon, MoreHorizontalIcon, PencilIcon } from 'lucide-react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SiHuggingface } from 'react-icons/si';

import { CpuOnlySetting, supportsCpuOnlySetting } from './CpuOnlySetting';
import { supportsDefaultSettings, type DefaultSettingsModel } from './defaultSettingsFields';
import { DefaultSettingsSection } from './DefaultSettingsSection';
import { MissingFileBadge, ModelBaseBadge, ModelFormatBadge } from './ModelBadges';
import { ModelEditForm } from './ModelEditForm';
import { ModelImageUpload } from './ModelImageUpload';
import { ModelSettingsMenuItems } from './ModelSettingsMenuItems';
import { RelatedModelsSection } from './RelatedModelsSection';
import { MemoizedTriggerPhrasesEditor } from './TriggerPhrasesEditor';
import { UpdatePathDialog } from './UpdatePathDialog';

const TRIGGER_PHRASE_TYPES = new Set(['main', 'lora', 'embedding']);
const EMPTY_TRIGGER_PHRASES: readonly string[] = [];

type ModelDetailShellModel = Pick<ModelConfig, 'key' | 'type'>;
type TriggerPhrasesModel = Pick<ModelConfig, 'key' | 'trigger_phrases'>;
type CpuOnlyModel = Pick<ModelConfig, 'cpu_only' | 'key' | 'name' | 'type'>;
type ModelIdentityModel = Pick<
  ModelConfig,
  | 'base'
  | 'config_path'
  | 'cover_image'
  | 'description'
  | 'file_size'
  | 'format'
  | 'hash'
  | 'image_encoder_model_id'
  | 'key'
  | 'name'
  | 'path'
  | 'prediction_type'
  | 'provider_id'
  | 'provider_model_id'
  | 'repo_variant'
  | 'source'
  | 'source_type'
  | 'source_url'
  | 'type'
  | 'variant'
>;

const findModel = (snapshot: ModelsSnapshot, modelKey: string): ModelConfig | undefined =>
  snapshot.modelsByKey.get(modelKey);

const selectModelShell = (snapshot: ModelsSnapshot, modelKey: string): ModelDetailShellModel | null => {
  const model = findModel(snapshot, modelKey);

  return model ? { key: model.key, type: model.type } : null;
};

const selectModelIdentity = (snapshot: ModelsSnapshot, modelKey: string): ModelIdentityModel | null => {
  const model = findModel(snapshot, modelKey);

  return model
    ? {
        base: model.base,
        config_path: model.config_path,
        cover_image: model.cover_image,
        description: model.description,
        file_size: model.file_size,
        format: model.format,
        hash: model.hash,
        image_encoder_model_id: model.image_encoder_model_id,
        key: model.key,
        name: model.name,
        path: model.path,
        prediction_type: model.prediction_type,
        provider_id: model.provider_id,
        provider_model_id: model.provider_model_id,
        repo_variant: model.repo_variant,
        source: model.source,
        source_type: model.source_type,
        source_url: model.source_url,
        type: model.type,
        variant: model.variant,
      }
    : null;
};

const selectDefaultSettingsModel = (snapshot: ModelsSnapshot, modelKey: string): DefaultSettingsModel | null => {
  const model = findModel(snapshot, modelKey);

  // `base` is projected too: the FP8 storage default is unavailable for Z-Image.
  return model
    ? { base: model.base, default_settings: model.default_settings, key: model.key, type: model.type }
    : null;
};

const selectTriggerPhrasesModel = (snapshot: ModelsSnapshot, modelKey: string): TriggerPhrasesModel | null => {
  const model = findModel(snapshot, modelKey);

  return model ? { key: model.key, trigger_phrases: model.trigger_phrases } : null;
};

const selectCpuOnlyModel = (snapshot: ModelsSnapshot, modelKey: string): CpuOnlyModel | null => {
  const model = findModel(snapshot, modelKey);

  return model ? { cpu_only: model.cpu_only, key: model.key, name: model.name, type: model.type } : null;
};

const areTriggerPhrasesModelsEqual = (left: TriggerPhrasesModel | null, right: TriggerPhrasesModel | null): boolean =>
  left?.key === right?.key && areArraysEqual(left?.trigger_phrases ?? [], right?.trigger_phrases ?? []);

/**
 * Full detail pane for one model: identity (view/edit), per-model default
 * settings, related models, trigger phrases, and lifecycle actions (convert,
 * re-identify, delete). Mount keyed by model key so per-model form state never
 * leaks between models.
 */
export const ModelDetail = ({ modelKey, onDeleted }: { modelKey: string; onDeleted: () => void }) => {
  const { t } = useTranslation();
  const model = useModelsSelector((snapshot) => selectModelShell(snapshot, modelKey));

  if (!model) {
    return (
      <Stack align="start" gap="2" p="1">
        <Text color="fg.subtle" fontSize="xs">
          {t('models.modelNoLongerInLibrary')}
        </Text>
      </Stack>
    );
  }

  return (
    <Stack gap="4" pb="4">
      <ModelIdentitySectionContainer modelKey={model.key} onDeleted={onDeleted} />

      {supportsCpuOnlySetting(model) ? (
        <>
          <Separator borderColor="border.subtle" />
          <CpuOnlySettingContainer modelKey={model.key} />
        </>
      ) : null}

      {supportsDefaultSettings(model) ? (
        <>
          <Separator borderColor="border.subtle" />
          <DefaultSettingsSectionContainer modelKey={model.key} />
        </>
      ) : null}

      {isLinkableType(model.type) ? (
        <>
          <Separator borderColor="border.subtle" />
          <RelatedModelsSectionContainer modelKey={model.key} />
        </>
      ) : null}

      {TRIGGER_PHRASE_TYPES.has(model.type) ? (
        <>
          <Separator borderColor="border.subtle" />
          <TriggerPhrasesEditorContainer modelKey={model.key} />
        </>
      ) : null}
    </Stack>
  );
};

interface ModelIdentitySectionProps {
  isMissing: boolean;
  model: ModelIdentityModel;
  onDeleted: () => void;
}

const ModelIdentitySectionContainer = memo(function ModelIdentitySectionContainer({
  modelKey,
  onDeleted,
}: {
  modelKey: string;
  onDeleted: () => void;
}) {
  const model = useModelsSelector((snapshot) => selectModelIdentity(snapshot, modelKey));
  const isMissing = useModelsSelector((snapshot) => snapshot.missingModelKeys.has(modelKey));

  if (!model) {
    return null;
  }

  return <ModelIdentitySection isMissing={isMissing} model={model} onDeleted={onDeleted} />;
});

const ModelIdentitySection = memo(function ModelIdentitySection({
  isMissing,
  model,
  onDeleted,
}: ModelIdentitySectionProps) {
  const notify = useNotify();
  const { t } = useTranslation();
  const [editingModelKey, setEditingModelKey] = useState<string | null>(null);
  const isEditing = editingModelKey === model.key;

  return (
    <>
      <HStack align="start" gap="3">
        <ModelImageUpload
          key={model.key}
          model={model}
          onError={(message) => notify.error(t('models.modelImage'), message)}
          onUpdated={() => notify.success(t('models.modelImageUpdated'))}
        />
        <Stack flex="1" gap="1" minW="0">
          <Text fontSize="sm" fontWeight="700" lineClamp={2}>
            {model.name}
          </Text>
          <HStack gap="1" minW="0" wrap="wrap">
            <ModelBaseBadge base={model.base} />
            <ModelFormatBadge format={model.format} />
          </HStack>
          {isMissing ? (
            <HStack gap="1.5">
              <MissingFileBadge />
              <Text color="fg.subtle" fontSize="2xs">
                {t('models.fileNotFoundOnDisk')}
              </Text>
            </HStack>
          ) : null}
          {model.description ? (
            <Text color="fg.muted" fontSize="2xs" lineClamp={3}>
              {model.description}
            </Text>
          ) : null}
        </Stack>
        <ModelDetailActions
          isEditing={isEditing}
          model={model}
          onDeleted={onDeleted}
          onToggleEditing={() => setEditingModelKey((key) => (key === model.key ? null : model.key))}
        />
      </HStack>

      {isEditing ? (
        <ModelEditForm
          model={model}
          onCancel={() => setEditingModelKey(null)}
          onSaved={() => {
            setEditingModelKey(null);
            notify.success(t('models.modelUpdated'), model.name);
          }}
        />
      ) : (
        <ModelAttributes isMissing={isMissing} model={model} />
      )}
    </>
  );
});

const ModelDetailActions = ({
  isEditing,
  model,
  onDeleted,
  onToggleEditing,
}: {
  isEditing: boolean;
  model: ModelIdentityModel;
  onDeleted: () => void;
  onToggleEditing: () => void;
}) => {
  const { t } = useTranslation();
  const [pendingAction, setPendingAction] = useState<PendingModelAction>(null);
  const [isActionBusy, setIsActionBusy] = useState(false);

  return (
    <HStack flexShrink={0} gap="1" wrap="wrap">
      {isConvertibleToDiffusers(model) ? (
        <Button size="xs" variant="outline" onClick={() => setPendingAction({ kind: 'convert', model })}>
          <Icon as={SiHuggingface} boxSize="3" />
          {t('models.convertToDiffusers')}
        </Button>
      ) : null}
      <Button size="xs" variant={isEditing ? 'solid' : 'outline'} onClick={onToggleEditing}>
        <Icon as={PencilIcon} boxSize="3" />
        {isEditing ? t('models.editing') : t('common.edit')}
      </Button>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
        <Menu.Trigger asChild>
          <IconButton aria-label={t('models.actions')} loading={isActionBusy} size="xs" variant="ghost">
            <Icon as={MoreHorizontalIcon} boxSize="4" />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="12rem">
              <ModelActionMenuItems
                extraItems={<ModelSettingsMenuItems modelKey={model.key} />}
                model={model}
                onBusyChange={setIsActionBusy}
                onRequestConfirm={setPendingAction}
              />
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ModelActionConfirmDialog pending={pendingAction} onClose={() => setPendingAction(null)} onDeleted={onDeleted} />
    </HStack>
  );
};

const DefaultSettingsSectionContainer = memo(function DefaultSettingsSectionContainer({
  modelKey,
}: {
  modelKey: string;
}) {
  const notify = useNotify();
  const { t } = useTranslation();
  const model = useModelsSelector((snapshot) => selectDefaultSettingsModel(snapshot, modelKey));

  if (!model || !supportsDefaultSettings(model)) {
    return null;
  }

  return (
    <DefaultSettingsSection
      model={model}
      onError={(message) => notify.error(t('models.defaultSettings'), message)}
      onSaved={() => notify.success(t('models.defaultSettingsSaved'))}
    />
  );
});

const CpuOnlySettingContainer = memo(function CpuOnlySettingContainer({ modelKey }: { modelKey: string }) {
  const notify = useNotify();
  const { t } = useTranslation();
  const model = useModelsSelector((snapshot) => selectCpuOnlyModel(snapshot, modelKey));

  if (!model || !supportsCpuOnlySetting(model)) {
    return null;
  }

  return (
    <CpuOnlySetting
      model={model}
      onError={(message) => notify.error(t('models.failedToSaveCpuSetting'), message)}
      onSaved={() => notify.success(t('models.cpuSettingSaved'), model.name)}
    />
  );
});

const TriggerPhrasesEditorContainer = memo(function TriggerPhrasesEditorContainer({ modelKey }: { modelKey: string }) {
  const notify = useNotify();
  const { t } = useTranslation();
  const handleError = useCallback((message: string) => notify.error(t('models.triggerPhrases'), message), [notify, t]);
  const model = useModelsSelector(
    (snapshot) => selectTriggerPhrasesModel(snapshot, modelKey),
    areTriggerPhrasesModelsEqual
  );

  if (!model) {
    return null;
  }

  return (
    <MemoizedTriggerPhrasesEditor
      modelKey={model.key}
      phrases={model.trigger_phrases ?? EMPTY_TRIGGER_PHRASES}
      onError={handleError}
    />
  );
});

const RelatedModelsSectionContainer = memo(function RelatedModelsSectionContainer({ modelKey }: { modelKey: string }) {
  const notify = useNotify();
  const { t } = useTranslation();
  const model = useModelsSelector(
    (snapshot) => {
      const candidate = snapshot.modelsByKey.get(modelKey);

      return candidate ? { base: candidate.base, key: candidate.key, type: candidate.type } : null;
    },
    (left, right) => left?.key === right?.key && left?.base === right?.base && left?.type === right?.type
  );
  const handleSectionError = useCallback(
    (message: string) => notify.error(t('models.modelManager'), message),
    [notify, t]
  );

  if (!model) {
    return null;
  }

  return <RelatedModelsSection model={model} onError={handleSectionError} />;
});

const ModelAttributes = ({ isMissing, model }: { isMissing: boolean; model: ModelIdentityModel }) => {
  const { t } = useTranslation();
  const modelsDir = useModelsSelector((snapshot) => snapshot.modelsDir);
  const [isPathDialogOpen, setIsPathDialogOpen] = useState(false);
  // Managed models store paths relative to the models directory; show the
  // resolved absolute path so it can be found on disk.
  const fullPath = resolveModelAbsolutePath(model.path, modelsDir);
  // In-place installs (absolute paths) may be repointed after the file moves;
  // a missing model gets the affordance too — that is exactly when it helps.
  const canUpdatePath = isAbsoluteModelPath(model.path) || isMissing;

  const attributes: { action?: ReactNode; href?: string; label: string; value: string }[] = [
    { label: t('models.fileSize'), value: formatBytes(model.file_size) },
    { label: t('models.variant'), value: model.variant ?? '—' },
    { label: t('models.predictionType'), value: model.prediction_type ?? '—' },
    { label: t('models.hash'), value: model.hash },
    {
      action: canUpdatePath ? (
        <IconButton
          aria-label={t('models.updatePath')}
          size="2xs"
          variant="ghost"
          onClick={() => setIsPathDialogOpen(true)}
        >
          <Icon as={PencilIcon} boxSize="3" />
        </IconButton>
      ) : undefined,
      label: t('models.path'),
      value: fullPath,
    },
    {
      href: getModelSourceHref(model.source, model.source_type) ?? undefined,
      label: t('models.source'),
      value: model.source,
    },
    // The user-editable page link (e.g. a Civitai listing); only visible in
    // the edit form until now. Old records may predate the http(s)
    // validation, so unlinkable values still render as text.
    ...(model.source_url
      ? [
          {
            href: model.source_url.startsWith('http') ? model.source_url : undefined,
            label: t('models.sourceUrl'),
            value: model.source_url,
          },
        ]
      : []),
    // Format-specific attrs; truthiness also skips repo_variant's '' default.
    ...(model.format === 'diffusers' && model.repo_variant
      ? [{ label: t('models.repoVariant'), value: model.repo_variant }]
      : []),
    ...(model.format === 'checkpoint' && model.config_path
      ? [{ label: t('models.configPath'), value: model.config_path }]
      : []),
    ...(model.image_encoder_model_id
      ? [{ label: t('models.imageEncoderModelId'), value: model.image_encoder_model_id }]
      : []),
    ...(model.provider_id ? [{ label: t('models.providerId'), value: model.provider_id }] : []),
    ...(model.provider_model_id ? [{ label: t('models.providerModelId'), value: model.provider_model_id }] : []),
  ];

  return (
    <>
      <DataList.Root gap="2.5" orientation="horizontal" size="sm" variant="subtle">
        {attributes.map((attribute) => (
          <DataList.Item key={attribute.label}>
            <DataList.ItemLabel color="fg.subtle" fontSize="2xs" minW="8rem" textTransform="uppercase">
              {attribute.label}
            </DataList.ItemLabel>
            <DataList.ItemValue alignItems="center" display="flex" fontSize="2xs" gap="1" overflowWrap="anywhere">
              {attribute.href ? (
                <chakra.a
                  alignItems="center"
                  display="inline-flex"
                  gap="1"
                  href={attribute.href}
                  minW="0"
                  rel="noreferrer"
                  target="_blank"
                  wordBreak="break-all"
                  _hover={{ textDecoration: 'underline' }}
                >
                  {attribute.value}
                  <Icon as={ExternalLinkIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
                </chakra.a>
              ) : (
                attribute.value
              )}
              {attribute.action ?? null}
            </DataList.ItemValue>
          </DataList.Item>
        ))}
      </DataList.Root>
      {isPathDialogOpen ? <UpdatePathDialog model={model} onClose={() => setIsPathDialogOpen(false)} /> : null}
    </>
  );
};
