/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ModelConfig, ModelIdentification } from '@features/models/core/types';
import type { ReactNode } from 'react';

import { HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { getIdentityChanges, getProposedModelLabel, shouldAutoIdentify } from '@features/models/core/identification';
import { modelIdentificationQueryOptions } from '@features/models/data/identificationQueries';
import { useModelActions } from '@features/models/ui/detail/useModelActions';
import { useModelsUi } from '@features/models/ui/ModelsUiContext';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { useQuery } from '@tanstack/react-query';
import { CircleCheckIcon, RefreshCcwIcon, SearchCheckIcon, TriangleAlertIcon, XIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

export type IdentificationNoticeModel = Pick<
  ModelConfig,
  'base' | 'format' | 'key' | 'name' | 'path' | 'type' | 'variant'
>;

const FIELD_LABEL_KEYS = {
  base: 'models.identificationFieldBase',
  format: 'models.identificationFieldFormat',
  type: 'models.identificationFieldType',
  variant: 'models.identificationFieldVariant',
} as const;

const NoticeFrame = ({
  actions,
  children,
  tone = 'warning',
}: {
  actions?: ReactNode;
  children: ReactNode;
  tone?: 'info' | 'warning';
}) => (
  <HStack
    align="start"
    bg="bg.subtle"
    borderColor="border.subtle"
    borderRadius="md"
    borderWidth="1px"
    gap="2"
    p="2.5"
    wrap="wrap"
  >
    <Icon
      as={tone === 'info' ? SearchCheckIcon : TriangleAlertIcon}
      boxSize="3.5"
      color={tone === 'info' ? 'fg.muted' : 'fg.warning'}
      flexShrink={0}
      mt="0.5"
    />
    <Stack flex="1 1 14rem" gap="1" minW="0" role="status">
      {children}
    </Stack>
    {actions ? (
      <HStack flexShrink={0} gap="1" ms="auto">
        {actions}
      </HStack>
    ) : null}
  </HStack>
);

/**
 * Tells an administrator when the model's file reads as something other than how it is filed, and applies the
 * existing re-identify on request. Unknown records and safetensors/GGUF files are checked on opening; other
 * formats read the whole file to identify it, so they wait for the button. Everyone else sees nothing.
 */
export const ModelIdentificationNotice = ({ model }: { model: IdentificationNoticeModel }) => {
  const { canManageModels } = useModelsUi();

  return canManageModels ? <IdentificationCheck model={model} /> : null;
};

const IdentificationCheck = ({ model }: { model: IdentificationNoticeModel }) => {
  const { t } = useTranslation();
  const { reidentify } = useModelActions();
  const isAutomatic = shouldAutoIdentify(model);
  const [isRequested, setIsRequested] = useState(false);
  // What the user dismissed, so a different proposal (after an edit, say) is not hidden with it.
  const [dismissedKey, setDismissedKey] = useState<string | null>(null);
  // The identity the user applied: the record then reads as the proposal by definition, so it is not probed again.
  const [appliedFingerprint, setAppliedFingerprint] = useState<string | null>(null);
  const [isApplying, setIsApplying] = useState(false);
  const fingerprint = identityFingerprint(model);
  const isApplied = appliedFingerprint === fingerprint;
  const query = useQuery({
    ...modelIdentificationQueryOptions(model.key, fingerprint),
    enabled: (isAutomatic || isRequested) && !isApplied,
  });

  const handleApply = async (proposed: IdentificationNoticeModel) => {
    setIsApplying(true);
    try {
      if (await reidentify(model)) {
        setAppliedFingerprint(identityFingerprint({ ...proposed, path: model.path }));
      }
    } finally {
      setIsApplying(false);
    }
  };

  if (isApplied) {
    return isRequested ? <MatchesNotice /> : null;
  }

  if (!isAutomatic && !isRequested) {
    return (
      <NoticeFrame
        actions={
          <Button variant="outline" onClick={() => setIsRequested(true)}>
            <Icon as={SearchCheckIcon} boxSize="3" />
            {t('models.identificationCheck')}
          </Button>
        }
        tone="info"
      >
        <Text color="fg.muted" fontSize="xs">
          {t('models.identificationCheckHint')}
        </Text>
      </NoticeFrame>
    );
  }

  if (query.isPending) {
    // Automatic checks stay out of the way; one the user asked for says it is working.
    return isRequested ? (
      <NoticeFrame tone="info">
        <HStack gap="2">
          <Spinner size="md" />
          <Text color="fg.muted" fontSize="xs">
            {t('models.identificationChecking')}
          </Text>
        </HStack>
      </NoticeFrame>
    ) : null;
  }

  if (query.isError) {
    return isRequested ? (
      <NoticeFrame
        actions={
          <Button variant="outline" onClick={() => void query.refetch()}>
            <Icon as={RefreshCcwIcon} boxSize="3" />
            {t('common.retry')}
          </Button>
        }
      >
        <Text color="fg.muted" fontSize="xs">
          {t('models.identificationFailed')}
        </Text>
        <Text color="fg.subtle" fontSize="xs" overflowWrap="anywhere">
          {getApiErrorMessage(query.error, '')}
        </Text>
      </NoticeFrame>
    ) : null;
  }

  const identification: ModelIdentification = query.data;

  if (identification.proposed === null) {
    const refusalKey = `${fingerprint}|refusal|${identification.refusal}`;

    if (identification.refusal === null || dismissedKey === refusalKey) {
      return null;
    }

    return (
      <NoticeFrame actions={<DismissButton onDismiss={() => setDismissedKey(refusalKey)} />}>
        <Text color="fg.muted" fontSize="xs">
          {t('models.identificationRefused', { reason: identification.refusal })}
        </Text>
      </NoticeFrame>
    );
  }

  if (!identification.changed) {
    return isRequested ? <MatchesNotice /> : null;
  }

  const { proposed } = identification;
  const proposalKey = `${fingerprint}|${identityFingerprint(proposed)}`;

  if (dismissedKey === proposalKey) {
    return null;
  }

  const changes = getIdentityChanges(model, proposed);

  return (
    <NoticeFrame
      actions={
        <>
          <Button loading={isApplying} variant="outline" onClick={() => void handleApply(proposed)}>
            {t('common.apply')}
          </Button>
          <DismissButton disabled={isApplying} onDismiss={() => setDismissedKey(proposalKey)} />
        </>
      }
    >
      <Text fontSize="xs" fontWeight="600">
        {t('models.identificationLooksLike', { label: getProposedModelLabel(proposed) })}
      </Text>
      <Stack as="ul" gap="0" listStyleType="none" m="0" p="0">
        {changes.map((change) => (
          <Text key={change.field} as="li" color="fg.muted" fontSize="xs">
            {t('models.identificationChange', {
              field: t(FIELD_LABEL_KEYS[change.field]),
              from: change.from ?? t('models.identificationNone'),
              to: change.to ?? t('models.identificationNone'),
            })}
          </Text>
        ))}
      </Stack>
    </NoticeFrame>
  );
};

const identityFingerprint = (model: Pick<ModelConfig, 'base' | 'format' | 'path' | 'type' | 'variant'>): string =>
  [model.type, model.format, model.base, model.variant ?? '', model.path].join('|');

const MatchesNotice = () => {
  const { t } = useTranslation();

  return (
    <NoticeFrame tone="info">
      <HStack gap="2">
        <Icon as={CircleCheckIcon} boxSize="3.5" color="fg.success" />
        <Text color="fg.muted" fontSize="xs">
          {t('models.identificationMatches')}
        </Text>
      </HStack>
    </NoticeFrame>
  );
};

const DismissButton = ({ disabled, onDismiss }: { disabled?: boolean; onDismiss: () => void }) => {
  const { t } = useTranslation();

  return (
    <Button disabled={disabled} variant="ghost" onClick={onDismiss}>
      <Icon as={XIcon} boxSize="3" />
      {t('models.identificationDismiss')}
    </Button>
  );
};
