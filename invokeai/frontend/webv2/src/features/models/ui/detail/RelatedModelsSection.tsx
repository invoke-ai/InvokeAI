/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig } from '@features/models/core/types';

import { Badge, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { getModelBaseColorPalette, getModelBaseLabel } from '@features/models/core/baseIdentity';
import { isBaseCompatible, LINKABLE_TYPES } from '@features/models/core/relationships';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import { useModelsSelector } from '@features/models/data/modelsStore';
import {
  linkModels,
  refreshRelatedModelKeys,
  unlinkModels,
  useRelatedModelKeys,
} from '@features/models/data/relationshipsStore';
import { ModelSelect } from '@features/models/ui/components';
import { SourceListItem } from '@features/models/ui/shared/SourceListItem';
import { useMountEffect } from '@platform/react/useMountEffect';
import { isAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { IconButton, FieldLabel, Tooltip } from '@platform/ui';
import { Link2OffIcon } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Bidirectional "related models" links — e.g. attach the LoRAs and VAE that
 * pair well with a checkpoint so other surfaces can suggest them together.
 * One grouped, compatibility-filtered picker searches every linkable type at
 * once; linked models render as rows with an unlink action.
 */
interface RelatedModelsSectionProps {
  model: Pick<ModelConfig, 'base' | 'key' | 'type'>;
  onError: (message: string) => void;
}

export const RelatedModelsSection = (props: RelatedModelsSectionProps) => (
  <RelatedModelsForModel key={props.model.key} {...props} />
);

const RelatedModelsForModel = ({ model, onError }: RelatedModelsSectionProps) => {
  const { t } = useTranslation();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const relatedKeys = useRelatedModelKeys(model.key);
  const [isMutating, setIsMutating] = useState(false);

  useMountEffect(() => {
    const owner = captureAccountScope();
    let isMounted = true;

    refreshRelatedModelKeys(model.key).catch((error: unknown) => {
      if (isMounted && isAccountScopeCurrent(owner)) {
        onError(error instanceof Error ? error.message : t('models.failedToLoadRelatedModels'));
      }
    });

    return () => {
      isMounted = false;
    };
  });

  const relatedModels = useMemo(() => {
    const byKey = new Map(models.map((candidate) => [candidate.key, candidate]));

    return (relatedKeys ?? []).map((key) => ({ key, model: byKey.get(key) ?? null }));
  }, [models, relatedKeys]);

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

      onError(error instanceof Error ? error.message : t('models.failedToLinkModels'));
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

      onError(error instanceof Error ? error.message : t('models.failedToUnlink'));
    } finally {
      if (isAccountScopeCurrent(owner)) {
        setIsMutating(false);
      }
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
        filter={(candidate) => isBaseCompatible(model, candidate)}
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
