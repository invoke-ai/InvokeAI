/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { RelatableModel } from '@features/models/core/relationships';
import type { ModelConfig } from '@features/models/core/types';
import type { ArchitectureCapabilitiesSnapshot } from '@features/models/data/architectureCapabilitiesStore';
import type { PickerStatus } from '@platform/ui/Picker';

import { Badge, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { getModelBaseColorPalette, getModelBaseLabel } from '@features/models/core/baseIdentity';
import {
  hasLinkableBase,
  isLinkablePair,
  isLinkableType,
  isVaePair,
  LINKABLE_TYPES,
} from '@features/models/core/relationships';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import {
  ensureArchitectureCapabilitiesLoaded,
  useArchitectureCapabilitiesSelector,
} from '@features/models/data/architectureCapabilitiesStore';
import { useModelsSelector } from '@features/models/data/modelsStore';
import {
  linkModels,
  refreshRelatedModelKeys,
  unlinkModels,
  useRelatedModelKeys,
} from '@features/models/data/relationshipsStore';
import { ModelSelect } from '@features/models/ui/components';
import { useMountEffect } from '@platform/react/useMountEffect';
import { isAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { IconButton, FieldLabel, Tooltip } from '@platform/ui';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { Link2OffIcon } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface RelatedModelsSectionProps {
  model: RelatableModel & Pick<ModelConfig, 'key'>;
  onError: (message: string) => void;
}

/** A new filter per table revision: the rule reads the table, and the compiler caches by arguments. */
const createLinkFilter =
  (subject: RelatableModel, _capabilitiesRevision: number) =>
  (candidate: ModelConfig): boolean =>
    isLinkablePair(subject, candidate);

/** Whether an installed, unlinked candidate waits on the architecture table to be decided. */
const hasUndecidedVaePairs = (
  subject: RelatableModel & Pick<ModelConfig, 'key'>,
  models: readonly ModelConfig[],
  excludeKeys: ReadonlySet<string>
): boolean =>
  models.some(
    (candidate) => !excludeKeys.has(candidate.key) && isLinkableType(candidate.type) && isVaePair(subject, candidate)
  );

const selectCapabilitiesRevision = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.revision;
const selectCapabilitiesStatus = (snapshot: ArchitectureCapabilitiesSnapshot) => snapshot.status;

export const RelatedModelsSection = (props: RelatedModelsSectionProps) => (
  <RelatedModelsForModel key={props.model.key} {...props} />
);

const RelatedModelsForModel = ({ model, onError }: RelatedModelsSectionProps) => {
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsByKey = useModelsSelector((snapshot) => snapshot.modelsByKey);
  const capabilitiesRevision = useArchitectureCapabilitiesSelector(selectCapabilitiesRevision);
  const capabilitiesStatus = useArchitectureCapabilitiesSelector(selectCapabilitiesStatus);
  const relatedKeys = useRelatedModelKeys(model.key);
  const [isMutating, setIsMutating] = useState(false);

  useMountEffect(() => {
    const owner = captureAccountScope();
    let isMounted = true;

    refreshRelatedModelKeys(model.key).catch((error: unknown) => {
      if (isMounted && isAccountScopeCurrent(owner)) {
        onError(getApiErrorMessage(error, t('models.failedToLoadRelatedModels')));
      }
    });

    return () => {
      isMounted = false;
    };
  });

  const relatedModels = useMemo(
    () => (relatedKeys ?? []).map((key) => ({ key, model: modelsByKey.get(key) ?? null })),
    [modelsByKey, relatedKeys]
  );

  const excludeKeys = useMemo(() => new Set([model.key, ...(relatedKeys ?? [])]), [model.key, relatedKeys]);

  const handleAdd = async (target: ModelConfig | null) => {
    if (!target || isMutating) {
      return;
    }

    const owner = captureAccountScope();

    setIsMutating(true);

    try {
      await linkModels(model.key, target.key);
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      onError(getApiErrorMessage(error, t('models.failedToLinkModels')));
    } finally {
      if (isAccountScopeCurrent(owner)) {
        setIsMutating(false);
      }
    }
  };

  const handleRemove = async (key: string) => {
    const owner = captureAccountScope();

    setIsMutating(true);

    try {
      await unlinkModels(model.key, key);
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      onError(getApiErrorMessage(error, t('models.failedToUnlink')));
    } finally {
      if (isAccountScopeCurrent(owner)) {
        setIsMutating(false);
      }
    }
  };

  // A subject whose base can never match a candidate (external/unknown, or an
  // any-based type with no allowance) keeps its persisted links manageable but
  // gets no add-picker — it would be a permanently empty combobox.
  const canAddLinks = hasLinkableBase(model);
  const linkFilter = createLinkFilter(model, capabilitiesRevision);
  // VAE pairs fail closed without the table; say so instead of offering a list that silently lacks them.
  const pendingVaePairs: PickerStatus | undefined =
    capabilitiesStatus === 'loaded' || !hasUndecidedVaePairs(model, models, excludeKeys)
      ? undefined
      : capabilitiesStatus === 'error'
        ? {
            isError: true,
            message: t('models.vaePairingLoadFailed'),
            onRetry: () => void ensureArchitectureCapabilitiesLoaded(),
          }
        : { message: t('models.vaePairingLoading') };

  return (
    <Stack gap="2">
      <Stack gap="0.5">
        <FieldLabel>{t('models.relatedModels')}</FieldLabel>
        {canAddLinks ? (
          <Text color="fg.subtle" fontSize="xs">
            {t('models.relatedModelsHelp')}
          </Text>
        ) : null}
      </Stack>
      {canAddLinks ? (
        <ModelSelect
          excludeKeys={excludeKeys}
          filter={linkFilter}
          modelTypes={LINKABLE_TYPES}
          pendingCandidates={pendingVaePairs}
          placeholder={t('models.searchCompatibleToLink')}
          showManagerButton={false}
          size="lg"
          value={null}
          onChange={(target) => {
            void handleAdd(target);
          }}
          onOpen={() => void ensureArchitectureCapabilitiesLoaded()}
        />
      ) : null}
      {relatedKeys === null ? (
        <Spinner color="fg.subtle" />
      ) : relatedModels.length === 0 ? (
        <Text color="fg.subtle" fontSize="xs">
          {t('models.noRelatedModels')}
        </Text>
      ) : (
        <ListStack label={t('models.relatedModels')}>
          {relatedModels.map(({ key, model: relatedModel }) => (
            <ListItem
              key={key}
              actions={
                <Tooltip content={t('models.unlink')}>
                  <IconButton
                    aria-label={t('models.unlinkNamed', { name: relatedModel?.name ?? key })}
                    disabled={isMutating}
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      void handleRemove(key);
                    }}
                  >
                    <Icon as={Link2OffIcon} boxSize="3" />
                  </IconButton>
                </Tooltip>
              }
              badges={
                relatedModel ? (
                  <>
                    <Badge
                      colorPalette={getModelBaseColorPalette(relatedModel.base)}
                      fontSize="xs"
                      size="lg"
                      variant="surface"
                    >
                      {getModelBaseLabel(relatedModel.base)}
                    </Badge>
                    <Badge colorPalette="gray" fontSize="xs" size="lg" variant="surface">
                      {getModelTypeLabel(relatedModel.type)}
                    </Badge>
                  </>
                ) : undefined
              }
              title={relatedModel?.name ?? key}
            />
          ))}
        </ListStack>
      )}
    </Stack>
  );
};
