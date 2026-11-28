/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ModelConfig, ModelIdentification } from '@features/models/core/types';
import type { ReactNode } from 'react';

import { HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { getIdentityChanges, getProposedModelLabel, shouldAutoIdentify } from '@features/models/core/identification';
import { modelIdentificationQueryOptions } from '@features/models/data/identificationQueries';
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

type RequestReidentify = (pending: { kind: 'reidentify'; model: IdentificationNoticeModel }) => void;

/**
 * Tells an administrator when the model's file reads as something other than how it is filed, and hands Apply to
 * the same confirmation as the other re-identify entry points. Unknown records and safetensors/GGUF files are
 * checked on opening; other formats wait for the button, because reading them runs a pickle scan. Everyone else
 * sees nothing.
 */
export const ModelIdentificationNotice = ({
  model,
  onRequestConfirm,
}: {
  model: IdentificationNoticeModel;
  onRequestConfirm: RequestReidentify;
}) => {
  const { canManageModels } = useModelsUi();

  return canManageModels ? <IdentificationCheck model={model} onRequestConfirm={onRequestConfirm} /> : null;
};

const IdentificationCheck = ({
  model,
  onRequestConfirm,
}: {
  model: IdentificationNoticeModel;
  onRequestConfirm: RequestReidentify;
}) => {
  const { t } = useTranslation();
  const isAutomatic = shouldAutoIdentify(model);
  const fingerprint = identityFingerprint(model);
  // The record the user asked to check. A later change to the record is a different file state, so it asks again.
  const [requestedFingerprint, setRequestedFingerprint] = useState<string | null>(null);
  const isRequested = requestedFingerprint === fingerprint;
  // What the user dismissed, so a different proposal (after an edit, say) is not hidden with it.
  const [dismissedKey, setDismissedKey] = useState<string | null>(null);
  const query = useQuery({
    ...modelIdentificationQueryOptions(model.key, fingerprint),
    enabled: isAutomatic || isRequested,
  });

  if (!isAutomatic && !isRequested) {
    return (
      <NoticeFrame
        actions={
          <Button variant="outline" onClick={() => setRequestedFingerprint(fingerprint)}>
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
          <Button variant="outline" onClick={() => onRequestConfirm({ kind: 'reidentify', model })}>
            {t('common.apply')}
          </Button>
          <DismissButton onDismiss={() => setDismissedKey(proposalKey)} />
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

const DismissButton = ({ onDismiss }: { onDismiss: () => void }) => {
  const { t } = useTranslation();

  return (
    <Button variant="ghost" onClick={onDismiss}>
      <Icon as={XIcon} boxSize="3" />
      {t('models.identificationDismiss')}
    </Button>
  );
};
