/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { AnyModelDefaultSettings, ModelConfig, ModelTaxonomyType } from '@workbench/models/types';

import { DataList, HStack, Icon, Menu, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton, ConfirmDialog, MenuContent } from '@workbench/components/ui';
import { isConvertibleToDiffusers } from '@workbench/models/baseIdentity';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { formatBytes } from '@workbench/models/taxonomy';
import { useNotify } from '@workbench/useNotify';
import { areArraysEqual } from '@workbench/workbenchSelectors';
import { ArrowLeftIcon, MoreHorizontalIcon, PencilIcon, RefreshCcwIcon, Trash2Icon } from 'lucide-react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SiHuggingface } from 'react-icons/si';

import { DefaultSettingsSection, supportsDefaultSettings } from './DefaultSettingsSection';
import { MissingFileBadge, ModelBaseBadge, ModelFormatBadge } from './ModelBadges';
import { ModelEditForm } from './ModelEditForm';
import { ModelImageUpload } from './ModelImageUpload';
import { RelatedModelsSection } from './RelatedModelsSection';
import { MemoizedTriggerPhrasesEditor } from './TriggerPhrasesEditor';
import { useModelActions } from './useModelActions';

const TRIGGER_PHRASE_TYPES = new Set(['main', 'lora', 'embedding']);
const RELATED_MODEL_TYPES = new Set(['main', 'lora']);
const EMPTY_TRIGGER_PHRASES: readonly string[] = [];

type ModelDetailShellModel = Pick<ModelConfig, 'key' | 'type'>;
type DefaultSettingsModel = { default_settings?: AnyModelDefaultSettings | null; key: string; type: ModelTaxonomyType };
type TriggerPhrasesModel = Pick<ModelConfig, 'key' | 'trigger_phrases'>;
type ModelIdentityModel = Pick<
  ModelConfig,
  | 'base'
  | 'cover_image'
  | 'description'
  | 'file_size'
  | 'format'
  | 'hash'
  | 'key'
  | 'name'
  | 'path'
  | 'prediction_type'
  | 'source'
  | 'source_url'
  | 'type'
  | 'variant'
>;

const findModel = (models: readonly ModelConfig[], modelKey: string): ModelConfig | undefined =>
  models.find((candidate) => candidate.key === modelKey);

const selectModelShell = (models: readonly ModelConfig[], modelKey: string): ModelDetailShellModel | null => {
  const model = findModel(models, modelKey);

  return model ? { key: model.key, type: model.type } : null;
};

const selectModelIdentity = (models: readonly ModelConfig[], modelKey: string): ModelIdentityModel | null => {
  const model = findModel(models, modelKey);

  return model
    ? {
        base: model.base,
        cover_image: model.cover_image,
        description: model.description,
        file_size: model.file_size,
        format: model.format,
        hash: model.hash,
        key: model.key,
        name: model.name,
        path: model.path,
        prediction_type: model.prediction_type,
        source: model.source,
        source_url: model.source_url,
        type: model.type,
        variant: model.variant,
      }
    : null;
};

const selectDefaultSettingsModel = (models: readonly ModelConfig[], modelKey: string): DefaultSettingsModel | null => {
  const model = findModel(models, modelKey);

  return model ? { default_settings: model.default_settings, key: model.key, type: model.type } : null;
};

const selectTriggerPhrasesModel = (models: readonly ModelConfig[], modelKey: string): TriggerPhrasesModel | null => {
  const model = findModel(models, modelKey);

  return model ? { key: model.key, trigger_phrases: model.trigger_phrases } : null;
};

const areTriggerPhrasesModelsEqual = (left: TriggerPhrasesModel | null, right: TriggerPhrasesModel | null): boolean =>
  left?.key === right?.key && areArraysEqual(left?.trigger_phrases ?? [], right?.trigger_phrases ?? []);

/**
 * Full detail pane for one model: identity (view/edit), per-model default
 * settings, related models, trigger phrases, and lifecycle actions (convert,
 * re-identify, delete). Mount keyed by model key so per-model form state never
 * leaks between models. `density="panel"` tightens the side-panel drill-in.
 */
export const ModelDetail = ({
  density = 'full',
  modelKey,
  onBack,
  onDeleted,
}: {
  density?: 'panel' | 'full';
  modelKey: string;
  onBack?: () => void;
  onDeleted: () => void;
}) => {
  const model = useModelsSelector((snapshot) => selectModelShell(snapshot.models, modelKey));

  if (!model) {
    return (
      <Stack align="start" gap="2" p="1">
        {onBack ? <BackButton onBack={onBack} /> : null}
        <Text color="fg.subtle" fontSize="xs">
          This model is no longer in the library.
        </Text>
      </Stack>
    );
  }

  return (
    <Stack gap={density === 'panel' ? '3' : '4'} pb="4">
      {onBack ? <BackButton onBack={onBack} /> : null}

      <ModelIdentitySectionContainer density={density} modelKey={model.key} onDeleted={onDeleted} />

      {supportsDefaultSettings(model) ? (
        <>
          <Separator borderColor="border.subtle" />
          <DefaultSettingsSectionContainer modelKey={model.key} />
        </>
      ) : null}

      {RELATED_MODEL_TYPES.has(model.type) ? (
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
  density: 'panel' | 'full';
  isMissing: boolean;
  model: ModelIdentityModel;
  onDeleted: () => void;
}

const ModelIdentitySectionContainer = memo(function ModelIdentitySectionContainer({
  density,
  modelKey,
  onDeleted,
}: {
  density: 'panel' | 'full';
  modelKey: string;
  onDeleted: () => void;
}) {
  const model = useModelsSelector((snapshot) => selectModelIdentity(snapshot.models, modelKey));
  const isMissing = useModelsSelector((snapshot) => snapshot.missingModelKeys.has(modelKey));

  if (!model) {
    return null;
  }

  return <ModelIdentitySection density={density} isMissing={isMissing} model={model} onDeleted={onDeleted} />;
});

const ModelIdentitySection = memo(function ModelIdentitySection({
  density,
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
        <ModelAttributes density={density} model={model} />
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
  const { convert, reidentify, remove } = useModelActions();
  const [pendingAction, setPendingAction] = useState<'delete' | 'convert' | null>(null);
  const [isActionBusy, setIsActionBusy] = useState(false);

  const handleReidentify = async () => {
    setIsActionBusy(true);

    try {
      await reidentify(model);
    } finally {
      setIsActionBusy(false);
    }
  };

  return (
    <HStack flexShrink={0} gap="1" wrap="wrap">
      {isConvertibleToDiffusers(model) ? (
        <Button size="xs" variant="outline" onClick={() => setPendingAction('convert')}>
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
              <Menu.Item value="reidentify" onClick={() => void handleReidentify()}>
                <Icon as={RefreshCcwIcon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">{t('models.reidentify')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Separator />
              <Menu.Item color="fg.error" value="delete" onClick={() => setPendingAction('delete')}>
                <Icon as={Trash2Icon} boxSize="3.5" />
                <Menu.ItemText fontSize="xs">{t('models.deleteModel')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <ConfirmDialog
        body={t('models.deleteBody', { name: model.name })}
        confirmLabel={t('models.deleteModel')}
        isOpen={pendingAction === 'delete'}
        title={t('models.deleteModel')}
        onClose={() => setPendingAction(null)}
        onConfirm={async () => {
          await remove(model);
          onDeleted();
        }}
      />
      {/* Destructive styling: the original checkpoint file is replaced. */}
      <ConfirmDialog
        body={t('models.convertBody', { name: model.name })}
        confirmLabel={t('models.convert')}
        isOpen={pendingAction === 'convert'}
        title={t('models.convertToDiffusers')}
        onClose={() => setPendingAction(null)}
        onConfirm={() => convert(model)}
      />
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
  const model = useModelsSelector((snapshot) => selectDefaultSettingsModel(snapshot.models, modelKey));

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

const TriggerPhrasesEditorContainer = memo(function TriggerPhrasesEditorContainer({ modelKey }: { modelKey: string }) {
  const notify = useNotify();
  const { t } = useTranslation();
  const handleError = useCallback((message: string) => notify.error(t('models.triggerPhrases'), message), [notify, t]);
  const model = useModelsSelector(
    (snapshot) => selectTriggerPhrasesModel(snapshot.models, modelKey),
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
      const candidate = snapshot.models.find((model) => model.key === modelKey);

      return candidate ? { base: candidate.base, key: candidate.key } : null;
    },
    (left, right) => left?.key === right?.key && left?.base === right?.base
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

const BackButton = ({ onBack }: { onBack: () => void }) => {
  const { t } = useTranslation();

  return (
    <Button alignSelf="start" size="2xs" variant="ghost" onClick={onBack}>
      <Icon as={ArrowLeftIcon} boxSize="3" />
      {t('models.allModels')}
    </Button>
  );
};

const isAbsolutePath = (path: string): boolean => path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path);

const ModelAttributes = ({ density, model }: { density: 'panel' | 'full'; model: ModelIdentityModel }) => {
  const { t } = useTranslation();
  const modelsDir = useModelsSelector((snapshot) => snapshot.modelsDir);
  // Managed models store paths relative to the models directory; show the
  // resolved absolute path so it can be found on disk.
  const fullPath =
    isAbsolutePath(model.path) || !modelsDir ? model.path : `${modelsDir.replace(/\/+$/, '')}/${model.path}`;

  const attributes: { label: string; value: string }[] = [
    { label: t('models.fileSize'), value: formatBytes(model.file_size) },
    { label: t('models.variant'), value: model.variant ?? '—' },
    { label: t('models.predictionType'), value: model.prediction_type ?? '—' },
    { label: t('models.hash'), value: model.hash },
    { label: t('models.path'), value: fullPath },
    { label: t('models.source'), value: model.source },
  ];

  return (
    <DataList.Root gap="2.5" orientation={density === 'panel' ? 'vertical' : 'horizontal'} size="sm" variant="subtle">
      {attributes.map((attribute) => (
        <DataList.Item key={attribute.label}>
          <DataList.ItemLabel color="fg.subtle" fontSize="2xs" minW="8rem" textTransform="uppercase">
            {attribute.label}
          </DataList.ItemLabel>
          <DataList.ItemValue fontSize="2xs" overflowWrap="anywhere">
            {attribute.value}
          </DataList.ItemValue>
        </DataList.Item>
      ))}
    </DataList.Root>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
