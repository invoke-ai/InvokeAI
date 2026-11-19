/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelConfig, ModelIdentificationOutcome } from '@features/models/core/types';

import { Checkbox, Flex, HStack, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { getProposedModelLabel } from '@features/models/core/identification';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import { bulkReidentifyModels } from '@features/models/data/api';
import { unknownModelScanQueryOptions } from '@features/models/data/identificationQueries';
import { getModelsSnapshot, refreshModels } from '@features/models/data/modelsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Panel } from '@platform/ui/Panel';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

interface Proposal {
  model: ModelConfig;
  proposed: ModelConfig;
}

interface Unresolved {
  key: string;
  model: ModelConfig | undefined;
  reason: string;
}

const splitOutcomes = (
  outcomes: readonly ModelIdentificationOutcome[],
  modelsByKey: ReadonlyMap<string, ModelConfig>,
  stillUnknown: string
): { proposals: Proposal[]; unresolved: Unresolved[] } => {
  const proposals: Proposal[] = [];
  const unresolved: Unresolved[] = [];

  for (const outcome of outcomes) {
    const model = modelsByKey.get(outcome.key);

    if (outcome.status === 'failed') {
      unresolved.push({ key: outcome.key, model, reason: outcome.error });
    } else if (model && outcome.identification.changed && outcome.identification.proposed) {
      proposals.push({ model, proposed: outcome.identification.proposed });
    } else {
      unresolved.push({ key: outcome.key, model, reason: outcome.identification.refusal ?? stillUnknown });
    }
  }

  return { proposals, unresolved };
};

/**
 * Previews what re-identifying every Unknown model would do and applies the ones the user keeps ticked. The models
 * are the ones Unknown when the dialog opened; the scan reads their files, so it runs once per opening.
 */
export const IdentifyUnknownModelsDialog = ({
  isOpen,
  onClose,
  onExitComplete,
}: {
  isOpen: boolean;
  onClose: () => void;
  /** After the close animation; the host releases the retained dialog here. */
  onExitComplete: () => void;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [snapshot] = useState(() => getModelsSnapshot());
  const [unknownKeys] = useState(() =>
    snapshot.models.filter((model) => model.type === 'unknown').map((model) => model.key)
  );
  // Everything starts ticked, so the user opts rows out instead of in.
  const [skipped, setSkipped] = useState<ReadonlySet<string>>(new Set());
  // Stopping ends the scan after the batch in flight; closing the dialog drops the query and abandons that too.
  const [stopScan] = useState(() => new AbortController());
  const scan = useQuery(unknownModelScanQueryOptions(unknownKeys, stopScan.signal));
  const checkedCount = scan.data?.length ?? 0;
  const isStopped = !scan.isFetching && !scan.isError && scan.data !== undefined && checkedCount < unknownKeys.length;
  const { isBusy: isApplying, run } = useScopedAction();

  const { proposals, unresolved } = scan.data
    ? splitOutcomes(scan.data, snapshot.modelsByKey, t('models.identifyUnknownStillUnknown'))
    : { proposals: [], unresolved: [] };
  const selectedKeys = proposals.map(({ model }) => model.key).filter((key) => !skipped.has(key));

  const toggle = (key: string) =>
    setSkipped((current) => {
      const next = new Set(current);

      if (!next.delete(key)) {
        next.add(key);
      }

      return next;
    });

  const handleApply = () =>
    run(
      async (owner) => {
        const result = await bulkReidentifyModels(selectedKeys, owner.signal);

        assertAccountScopeCurrent(owner);
        // The endpoint returns keys only; the refreshed library carries the new records.
        await refreshModels(owner);
        assertAccountScopeCurrent(owner);

        if (result.failed.length > 0) {
          notify.error(
            t('models.someCouldNotBeReidentified'),
            t('models.bulkReidentifyPartialDescription', {
              error: result.failed[0]?.error ?? '',
              failed: result.failed.length,
              succeeded: result.succeeded.length,
            })
          );
        } else {
          notify.success(
            t('models.reidentified'),
            t('models.reidentifiedDescription', { count: result.succeeded.length })
          );
        }

        onClose();
      },
      (message) => notify.error(t('models.bulkReidentifyFailed'), message)
    );

  return (
    <Dialog.Root
      open={isOpen}
      scrollBehavior="inside"
      size="md"
      onExitComplete={onExitComplete}
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
              <Stack gap="0.5">
                <Dialog.Title>{t('models.identifyUnknownTitle')}</Dialog.Title>
                <Dialog.Description>{t('models.identifyUnknownDescription')}</Dialog.Description>
              </Stack>
            </Dialog.Header>
            <Dialog.Body>
              {scan.isPending && !scan.isError ? (
                <Flex align="center" direction="column" gap="2" justify="center" py="8">
                  <Spinner color="fg.subtle" size="lg" />
                  <Text color="fg.muted" fontSize="xs">
                    {t('models.identifyUnknownChecking', { count: unknownKeys.length })}
                  </Text>
                </Flex>
              ) : (
                <Stack gap="3">
                  {scan.isError ? (
                    <Stack align="start" gap="2">
                      <Text color="fg.error" fontSize="md">
                        {t('models.identifyUnknownFailed', { error: getApiErrorMessage(scan.error, '') })}
                      </Text>
                      <Button variant="outline" onClick={() => void scan.refetch()}>
                        {t('common.retry')}
                      </Button>
                    </Stack>
                  ) : scan.isFetching ? (
                    <HStack gap="2" role="status">
                      <Spinner size="md" />
                      <Text color="fg.muted" flex="1" fontSize="xs">
                        {t('models.identifyUnknownProgress', { done: checkedCount, total: unknownKeys.length })}
                      </Text>
                      <Button size="sm" variant="ghost" onClick={() => stopScan.abort()}>
                        {t('models.identifyUnknownStop')}
                      </Button>
                    </HStack>
                  ) : isStopped ? (
                    <Text color="fg.muted" fontSize="xs" role="status">
                      {t('models.identifyUnknownStopped', { done: checkedCount, total: unknownKeys.length })}
                    </Text>
                  ) : null}
                  {proposals.length === 0 && !scan.isFetching ? (
                    <Text color="fg.muted" fontSize="md" py="2" textAlign="center">
                      {t('models.identifyUnknownNoneFound')}
                    </Text>
                  ) : (
                    <Stack gap="1.5">
                      <Text color="fg.muted" fontSize="xs" fontWeight="600">
                        {t('models.identifyUnknownCanBeIdentified', { count: proposals.length })}
                      </Text>
                      {proposals.map(({ model, proposed }) => (
                        <Panel key={model.key} alignItems="center" flexDirection="row" gap="2" p="2" tone="control">
                          <Checkbox.Root
                            checked={!skipped.has(model.key)}
                            colorPalette="accent"
                            disabled={isApplying}
                            onCheckedChange={() => toggle(model.key)}
                          >
                            <Checkbox.HiddenInput aria-label={model.name} />
                            <Checkbox.Control />
                          </Checkbox.Root>
                          <Stack flex="1" gap="0" minW="0">
                            <Text fontSize="xs" fontWeight="600" overflowWrap="anywhere">
                              {model.name}
                            </Text>
                            <Text color="fg.subtle" fontSize="xs">
                              {t('models.identifyUnknownBecomes', {
                                from: getModelTypeLabel(model.type),
                                to: getProposedModelLabel(proposed),
                              })}
                            </Text>
                          </Stack>
                        </Panel>
                      ))}
                    </Stack>
                  )}
                  {unresolved.length > 0 ? (
                    <Stack gap="1.5">
                      <Text color="fg.muted" fontSize="xs" fontWeight="600">
                        {t('models.identifyUnknownCannotBeIdentified', { count: unresolved.length })}
                      </Text>
                      {unresolved.map(({ key, model, reason }) => (
                        <Stack key={key} gap="0" px="2">
                          <Text fontSize="xs" overflowWrap="anywhere">
                            {model?.name ?? key}
                          </Text>
                          <Text color="fg.subtle" fontSize="xs" overflowWrap="anywhere">
                            {reason}
                          </Text>
                        </Stack>
                      ))}
                    </Stack>
                  ) : null}
                </Stack>
              )}
            </Dialog.Body>
            <Dialog.Footer>
              <Button disabled={isApplying} variant="ghost" onClick={onClose}>
                {t('common.close')}
              </Button>
              <Button
                disabled={selectedKeys.length === 0}
                loading={isApplying}
                variant="solid"
                onClick={() => void handleApply()}
              >
                {t('models.identifyUnknownApply', { count: selectedKeys.length })}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
