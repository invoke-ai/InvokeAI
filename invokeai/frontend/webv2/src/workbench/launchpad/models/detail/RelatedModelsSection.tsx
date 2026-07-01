/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig, ModelTaxonomyType } from '@workbench/models/types';

import { Badge, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { IconButton, FieldLabel, Tooltip } from '@workbench/components/ui';
import { SourceListItem } from '@workbench/launchpad/models/shared/SourceListItem';
import { addModelRelationship, getRelatedModelKeys, removeModelRelationship } from '@workbench/models/api';
import { getModelBaseColorPalette, getModelBaseLabel } from '@workbench/models/baseIdentity';
import { ModelSelect } from '@workbench/models/components';
import { useModelsSelector } from '@workbench/models/modelsStore';
import { getModelTypeLabel } from '@workbench/models/taxonomy';
import { Link2OffIcon } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Types offered when linking related models, grouped in one picker. */
const LINKABLE_TYPES: ModelTaxonomyType[] = [
  'main',
  'lora',
  'embedding',
  'vae',
  'controlnet',
  't2i_adapter',
  'ip_adapter',
];

/**
 * Bidirectional "related models" links — e.g. attach the LoRAs and VAE that
 * pair well with a checkpoint so other surfaces can suggest them together.
 * One grouped, compatibility-filtered picker searches every linkable type at
 * once; linked models render as rows with an unlink action.
 */
export const RelatedModelsSection = ({
  model,
  onError,
}: {
  model: Pick<ModelConfig, 'base' | 'key'>;
  onError: (message: string) => void;
}) => {
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const [relatedKeys, setRelatedKeys] = useState<string[] | null>(null);
  const [isMutating, setIsMutating] = useState(false);

  useEffect(() => {
    let isStale = false;

    getRelatedModelKeys(model.key)
      .then((keys) => {
        if (!isStale) {
          setRelatedKeys(keys);
        }
      })
      .catch((error: unknown) => {
        if (!isStale) {
          setRelatedKeys([]);
          onError(error instanceof Error ? error.message : t('models.failedToLoadRelatedModels'));
        }
      });

    return () => {
      isStale = true;
    };
  }, [model.key, onError, t]);

  const relatedModels = useMemo(() => {
    const byKey = new Map(models.map((candidate) => [candidate.key, candidate]));

    return (relatedKeys ?? []).map((key) => ({ key, model: byKey.get(key) ?? null }));
  }, [models, relatedKeys]);

  const excludeKeys = useMemo(() => new Set([model.key, ...(relatedKeys ?? [])]), [model.key, relatedKeys]);

  const handleAdd = async (target: ModelConfig | null) => {
    if (!target || isMutating) {
      return;
    }

    setIsMutating(true);

    try {
      await addModelRelationship(model.key, target.key);
      setRelatedKeys((current) => [...(current ?? []), target.key]);
    } catch (error) {
      onError(error instanceof Error ? error.message : t('models.failedToLinkModels'));
    } finally {
      setIsMutating(false);
    }
  };

  const handleRemove = async (key: string) => {
    setIsMutating(true);

    try {
      await removeModelRelationship(model.key, key);
      setRelatedKeys((current) => (current ?? []).filter((existing) => existing !== key));
    } catch (error) {
      onError(error instanceof Error ? error.message : t('models.failedToUnlink'));
    } finally {
      setIsMutating(false);
    }
  };

  return (
    <Stack gap="2">
      <Stack gap="0.5">
        <FieldLabel>{t('models.relatedModels')}</FieldLabel>
        <Text color="fg.subtle" fontSize="2xs">
          {t('models.relatedModelsHelp')}
        </Text>
      </Stack>
      <ModelSelect
        excludeKeys={excludeKeys}
        filter={(candidate) => candidate.base === model.base || candidate.base === 'any' || model.base === 'any'}
        modelTypes={LINKABLE_TYPES}
        placeholder={t('models.searchCompatibleToLink')}
        showManagerButton={false}
        size="sm"
        value={null}
        onChange={(target) => {
          void handleAdd(target);
        }}
      />
      {relatedKeys === null ? (
        <Spinner color="fg.subtle" size="xs" />
      ) : relatedModels.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('models.noRelatedModels')}
        </Text>
      ) : (
        <Stack gap="1.5">
          {relatedModels.map(({ key, model: relatedModel }) => (
            <SourceListItem
              key={key}
              badges={
                relatedModel ? (
                  <>
                    <Badge
                      colorPalette={getModelBaseColorPalette(relatedModel.base)}
                      flexShrink={0}
                      fontSize="2xs"
                      size="sm"
                      variant="surface"
                    >
                      {getModelBaseLabel(relatedModel.base)}
                    </Badge>
                    <Badge colorPalette="gray" flexShrink={0} fontSize="2xs" size="sm" variant="outline">
                      {getModelTypeLabel(relatedModel.type)}
                    </Badge>
                  </>
                ) : undefined
              }
              title={relatedModel?.name ?? key}
              trailing={
                <Tooltip content={t('models.unlink')}>
                  <IconButton
                    aria-label={t('models.unlinkNamed', { name: relatedModel?.name ?? key })}
                    disabled={isMutating}
                    size="2xs"
                    variant="ghost"
                    onClick={() => {
                      void handleRemove(key);
                    }}
                  >
                    <Icon as={Link2OffIcon} boxSize="3" />
                  </IconButton>
                </Tooltip>
              }
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
