import type { StarterInstallSource } from '@features/models';
import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowPreviewView } from '@features/workflow/ui/graph-preview/WorkflowPreviewStage';

import { Badge, Box, DataList, HStack, Menu, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import { getModelBaseLabel, getStarterModelInstallSources, useInstallActions } from '@features/models';
import {
  getLibraryWorkflowRecord,
  invalidateWorkflowLibraryCache,
  setLibraryWorkflowPublic,
} from '@features/workflow/queries';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useModelRequirementDeps } from '@features/workflow/ui/modelRequirementStatus';
import { useOpenAddModels, useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton, MenuContent, Scrollable, Tooltip, useTooltipTriggerIds } from '@platform/ui';
import { DownloadIcon, EllipsisIcon, WorkflowIcon } from 'lucide-react';
import { memo, useCallback, useLayoutEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowPreviewStepperProps } from './WorkflowPreviewStepper';

import { formatTimeSince, useDataTimeSnapshot } from './relativeTime';
import { useWorkflowLibraryEntryActions, type WorkflowLibraryDuplicateGuard } from './useWorkflowLibraryEntryActions';
import {
  useWorkflowPreviewModel,
  WORKFLOW_SILHOUETTE_THUMBNAIL_CSS,
  WorkflowAuthorNotes,
  WorkflowInputsBadge,
  WorkflowInputsSummary,
  WorkflowProvenance,
} from './WorkflowDetailOverview';
import {
  WORKFLOW_DETAIL_WIDE_ONLY_CSS,
  WORKFLOW_PREVIEW_TRIGGER_CSS,
  WorkflowDetailRailPlaceholder,
  WorkflowDetailSheet,
  WorkflowMissingNodePacks,
} from './WorkflowDetailSheet';
import { keepMenuOpenWhenSiblingMenuCloses } from './WorkflowLibraryCard';
import { WorkflowLibraryThumbnail } from './WorkflowLibraryThumbnail';
import { getMissingNodePacks, resolveEntryRequirements, WorkflowRequirementsList } from './WorkflowRequirementsList';

/** Kept mounted across selections to prevent flashing. */

const INSTALL_HOVER = { opacity: 0.85 } as const;
/** The large preview is wide-only; the expanded sheet's summary carries a small one. */
const THUMBNAIL_SLOT_CSS = { ...WORKFLOW_DETAIL_WIDE_ONLY_CSS, ...WORKFLOW_SILHOUETTE_THUMBNAIL_CSS };

type VisibilityState =
  | { requestId: object; previousItem: WorkflowLibraryListItem; isPublic: boolean; status: 'pending' | 'success' }
  | {
      requestId: object;
      previousItem: WorkflowLibraryListItem;
      isPublic: boolean;
      message: string;
      status: 'error' | 'unknown' | 'reconciling';
    };

export interface WorkflowLibraryDetailPanelProps {
  canChangeVisibility?: boolean;
  entry: WorkflowLibraryEntry | null;
  /** Shared with the card menus, so one workflow gets one copy per press. */
  duplicateGuard: WorkflowLibraryDuplicateGuard;
  /** Hides writes to bundled templates and to other accounts' (`getLibraryEntryAccess`). */
  isReadOnly: boolean;
  onClose: () => void;
  onDeleted: (workflowId: string) => void;
  onDuplicated: (workflowId: string) => void;
  onOpen: (item: WorkflowLibraryListItem) => void;
  /** Enters preview mode on the entry (on a given view, else the last shown), also while loading or unreadable. */
  onPreview: (workflowId: string, view?: WorkflowPreviewView) => void;
  projectWorkflows: readonly ProjectWorkflowEntry[];
  stepper?: WorkflowPreviewStepperProps | null;
}

export const WorkflowDetailFact = ({ children, label }: { children: ReactNode; label: string }) => (
  <DataList.Item alignItems="center" gap="2" minW="0">
    <DataList.ItemLabel color="fg.muted" flex="0 0 5.5rem" fontSize="xs" minW="0">
      {label}
    </DataList.ItemLabel>
    <DataList.ItemValue flex="1" fontSize="xs" minW="0" overflowWrap="anywhere">
      {children}
    </DataList.ItemValue>
  </DataList.Item>
);

/** Memoised: browsing re-renders the dialog on every range load and enrichment, rarely changing the selection. */
export const WorkflowLibraryDetailPanel = memo(function WorkflowLibraryDetailPanel({
  canChangeVisibility = false,
  duplicateGuard,
  entry,
  isReadOnly,
  onClose,
  onDeleted,
  onDuplicated,
  onOpen,
  onPreview,
  projectWorkflows,
  stepper,
}: WorkflowLibraryDetailPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  // Portaled action menus belong to this dialog's visible bounds, rather than the whole viewport.
  const menuPositioning = useMemo(
    () => ({
      boundary: () => panelRef.current?.closest<HTMLElement>('[data-scope="dialog"][data-part="content"]') ?? [],
    }),
    []
  );
  const { t } = useTranslation();
  const deps = useModelRequirementDeps();
  const notify = useWorkflowNotifications();
  const openAddModels = useOpenAddModels();
  const { installMany } = useInstallActions();
  // The rail's More and the sheet's More act on the selection; a card's own menu has its own instance.
  const {
    actionItems,
    dialogs,
    open: handleOpen,
    openLabel,
    projectCopies,
  } = useWorkflowLibraryEntryActions({
    duplicateGuard,
    entry,
    isReadOnly,
    projectWorkflows,
    onClose,
    onDeleted,
    onDuplicated,
    onOpen,
  });
  const [visibilityStates, setVisibilityStates] = useState<Map<string, VisibilityState>>(() => new Map());
  const visibilityInputRef = useRef<HTMLInputElement>(null);
  const restoreVisibilityFocus = useRef<string | null>(null);

  const enrichment = entry?.enrichment ?? null;
  // Shares the grid's per-entry cache, so selecting a card the badges already resolved costs nothing.
  const resolved = useMemo(
    () => (enrichment?.status === 'ready' ? resolveEntryRequirements(enrichment, deps) : null),
    [deps, enrichment]
  );
  const installableCount = resolved?.filter((requirement) => requirement.status === 'installable').length ?? 0;
  const session = useAuthSession();
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const missingNodePacks = useMemo(
    () =>
      enrichment?.status === 'ready' && templatesSnapshot.status === 'loaded'
        ? getMissingNodePacks(enrichment, templatesSnapshot.templates)
        : [],
    [enrichment, templatesSnapshot]
  );
  const dataTime = useDataTimeSnapshot(entry?.item);
  const moreActionsIds = useTooltipTriggerIds();

  const handlePreview = useCallback(() => entry && onPreview(entry.item.workflow_id), [entry, onPreview]);
  const handlePreviewInputs = useCallback(
    () => entry && onPreview(entry.item.workflow_id, 'inputs'),
    [entry, onPreview]
  );
  const workflowDocument = enrichment?.status === 'ready' ? enrichment.document : null;
  const previewModel = useWorkflowPreviewModel(workflowDocument);

  const handleFindModel = useCallback(
    (query: string) => {
      openAddModels(query);
      onClose();
    },
    [onClose, openAddModels]
  );

  const install = useCallback(async () => {
    const owner = captureAccountScope();
    const starterBySource = new Map(deps.starterModels.map((starter) => [starter.source, starter]));
    const requests: StarterInstallSource[] = [];
    const seen = new Set<string>();

    for (const requirement of resolved ?? []) {
      const starter = requirement.starterMatch ? starterBySource.get(requirement.starterMatch.source) : undefined;

      if (requirement.status !== 'installable' || !starter) {
        continue;
      }

      for (const source of getStarterModelInstallSources(starter)) {
        // Requirements often share a dependency, and an install may have started since this render.
        if (seen.has(source.source) || deps.activeInstallSources.has(source.source)) {
          continue;
        }

        seen.add(source.source);
        requests.push(source);
      }
    }

    if (requests.length === 0) {
      return;
    }

    const queued = await installMany(requests);

    // One notice for the whole set; `installMany` toasts its own failures.
    if (queued > 0 && isAccountScopeCurrent(owner)) {
      notify.success(t('workflowLibrary.installQueued'));
    }
  }, [deps, installMany, notify, resolved, t]);
  const handleInstall = useCallback(() => void install(), [install]);

  const item = entry?.item ?? null;
  const tags = entry?.tags ?? [];
  const name = item ? item.name || t('workflowLibrary.untitled') : '';
  const visibilityState = item ? visibilityStates.get(item.workflow_id) : undefined;
  const selectedWorkflowId = item?.workflow_id ?? null;
  const visibilityUnknown = visibilityState?.status === 'unknown' || visibilityState?.status === 'reconciling';
  const itemIsPublic = visibilityUnknown
    ? undefined
    : item && visibilityState?.previousItem === item
      ? visibilityState.isPublic
      : item?.is_public === true;
  const isVisibilityPending = visibilityState?.status === 'pending';
  const visibilityError = visibilityState?.status === 'error' || visibilityUnknown ? visibilityState.message : null;
  useLayoutEffect(() => {
    if (restoreVisibilityFocus.current !== null && restoreVisibilityFocus.current !== selectedWorkflowId) {
      restoreVisibilityFocus.current = null;
    }
    if (
      selectedWorkflowId &&
      restoreVisibilityFocus.current === selectedWorkflowId &&
      !isVisibilityPending &&
      !visibilityUnknown
    ) {
      restoreVisibilityFocus.current = null;
      visibilityInputRef.current?.focus();
    }
  }, [isVisibilityPending, itemIsPublic, selectedWorkflowId, visibilityUnknown]);
  const reconcileVisibility = useCallback(
    async (
      workflowId: string,
      previousItem: WorkflowLibraryListItem,
      requestId: object,
      owner: ReturnType<typeof captureAccountScope>,
      requestedVisibility?: boolean,
      failureMessage?: string
    ) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }
      invalidateWorkflowLibraryCache(workflowId);
      try {
        const record = await getLibraryWorkflowRecord(workflowId, owner.signal);
        assertAccountScopeCurrent(owner);
        const isPublic = record.is_public === true;
        setVisibilityStates((current) => {
          if (current.get(workflowId)?.requestId !== requestId) {
            return current;
          }
          const state = { isPublic, previousItem, requestId };
          return new Map(current).set(
            workflowId,
            requestedVisibility !== undefined && isPublic !== requestedVisibility
              ? { ...state, message: failureMessage ?? t('common.unknownError'), status: 'error' }
              : { ...state, status: 'success' }
          );
        });
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          setVisibilityStates((current) => {
            const state = current.get(workflowId);
            if (state?.requestId !== requestId) {
              return current;
            }
            return new Map(current).set(workflowId, {
              ...state,
              message: failureMessage ?? getApiErrorMessage(error, t('common.unknownError')),
              status: 'unknown',
            });
          });
        }
      }
    },
    [t]
  );
  const updateVisibility = useCallback(
    async (isPublic: boolean) => {
      if (!item || !canChangeVisibility || isVisibilityPending || visibilityUnknown || isPublic === itemIsPublic) {
        return;
      }

      const owner = captureAccountScope();
      const workflowId = item.workflow_id;
      const previousItem = item;
      const requestId = {};
      const previousVisibility = itemIsPublic === true;
      restoreVisibilityFocus.current = document.activeElement === visibilityInputRef.current ? workflowId : null;
      setVisibilityStates((current) =>
        new Map(current).set(workflowId, {
          isPublic: previousVisibility,
          previousItem,
          requestId,
          status: 'pending',
        })
      );

      try {
        const record = await setLibraryWorkflowPublic(workflowId, isPublic, owner.signal);

        assertAccountScopeCurrent(owner);
        setVisibilityStates((current) => {
          if (current.get(workflowId)?.requestId !== requestId) {
            return current;
          }
          return new Map(current).set(workflowId, {
            isPublic: record.is_public === true,
            previousItem,
            requestId,
            status: 'success',
          });
        });
        invalidateWorkflowLibraryCache(workflowId);
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }
        await reconcileVisibility(
          workflowId,
          previousItem,
          requestId,
          owner,
          isPublic,
          getApiErrorMessage(error, t('common.unknownError'))
        );
      } finally {
        setVisibilityStates((current) => {
          const request = current.get(workflowId);
          if (request?.requestId !== requestId || request.status !== 'pending') {
            return current;
          }
          const next = new Map(current);
          next.delete(workflowId);
          return next;
        });
      }
    },
    [canChangeVisibility, isVisibilityPending, item, itemIsPublic, reconcileVisibility, t, visibilityUnknown]
  );
  const retryVisibilityRead = useCallback(async () => {
    if (!item || !visibilityUnknown || !visibilityState || visibilityState.status === 'reconciling') {
      return;
    }
    const owner = captureAccountScope();
    const workflowId = item.workflow_id;
    const requestId = {};
    const previousItem = item;
    setVisibilityStates((current) =>
      new Map(current).set(workflowId, {
        ...visibilityState,
        previousItem,
        requestId,
        status: 'reconciling',
      })
    );
    await reconcileVisibility(workflowId, previousItem, requestId, owner);
  }, [item, reconcileVisibility, visibilityState, visibilityUnknown]);
  const handleRetryVisibilityRead = useCallback(() => void retryVisibilityRead(), [retryVisibilityRead]);
  const handleVisibilityClick = useCallback(
    (event: MouseEvent<HTMLInputElement>) => {
      event.preventDefault();
      void updateVisibility(!itemIsPublic);
    },
    [itemIsPublic, updateVisibility]
  );
  // Visibility only means something to other accounts on a multi-user server.
  const showVisibility = session.multiuserEnabled && item?.category === 'user';
  const primaryBase = enrichment?.status === 'ready' ? enrichment.requirements.primaryBase : null;
  const isCallable = item?.call_saved_workflow_compatibility?.is_callable === true;
  const linkedRevision = projectCopies.length === 1 ? (projectCopies[0]?.source?.revision ?? null) : null;
  const openedWhen = item?.opened_at ? formatTimeSince(item.opened_at, dataTime) : '';
  const updatedWhen = item?.updated_at ? formatTimeSince(item.updated_at, dataTime) : '';
  const createdWhen = item?.created_at ? formatTimeSince(item.created_at, dataTime) : '';

  const readiness =
    enrichment?.status !== 'ready'
      ? null
      : missingNodePacks.length > 0
        ? t('workflowLibrary.nodePacksMissing', { count: missingNodePacks.length })
        : installableCount > 0
          ? t('workflowLibrary.modelsToInstall', { count: installableCount })
          : t('workflowLibrary.readyToRun');
  const summaryFacts = [
    primaryBase ? getModelBaseLabel(primaryBase) : null,
    enrichment?.status === 'ready' ? t('workflowLibrary.nodeCount', { count: enrichment.nodeCount }) : null,
    readiness,
    openedWhen ? t('workflowLibrary.openedWhen', { when: openedWhen }) : null,
  ]
    .filter((fact) => fact !== null)
    .join(' · ');

  return (
    <>
      {item ? (
        <WorkflowDetailSheet
          ref={panelRef}
          actions={actionItems}
          data-workflow-detail={item.workflow_id}
          facts={summaryFacts}
          menuPositioning={menuPositioning}
          name={name}
          openLabel={openLabel}
          thumbnailSrc={item.thumbnail_url}
          workflowDocument={enrichment?.status === 'ready' ? enrichment.document : null}
          stepper={stepper}
          workflowId={item.workflow_id}
          onOpen={handleOpen}
        >
          <Scrollable flex="1" label={name} minH="0">
            <Stack gap="2.5" minW="0" p="2.5">
              {/* The expanded sheet's summary is its compact heading: no large preview, no second name. */}
              <Box css={THUMBNAIL_SLOT_CSS} w="full">
                <WorkflowLibraryThumbnail
                  key={item.workflow_id}
                  canEdit={!isReadOnly}
                  item={item}
                  workflowDocument={workflowDocument}
                  onPreview={handlePreview}
                />
              </Box>

              <Stack css={WORKFLOW_DETAIL_WIDE_ONLY_CSS} gap="0.5" minW="0">
                {/* Full names wrap, delimiter-free ones too; zero min-width contains them without truncating. */}
                <Text fontSize="lg" fontWeight="600" minW="0" overflowWrap="anywhere">
                  {name}
                </Text>
                <WorkflowProvenance
                  document={workflowDocument}
                  kind={t(
                    item.category === 'default'
                      ? 'workflowLibrary.preview.kindTemplate'
                      : isReadOnly
                        ? 'workflowLibrary.preview.kindShared'
                        : 'workflowLibrary.preview.kindLibrary'
                  )}
                />
                {openedWhen ? (
                  <Text color="fg.muted" fontSize="xs">
                    {t('workflowLibrary.openedWhen', { when: openedWhen })}
                  </Text>
                ) : null}
              </Stack>

              {primaryBase || enrichment?.status === 'ready' || isCallable || missingNodePacks.length > 0 ? (
                <HStack flexWrap="wrap" gap="1" minW="0">
                  {primaryBase ? <Badge variant="subtle">{getModelBaseLabel(primaryBase)}</Badge> : null}
                  {enrichment?.status === 'ready' ? (
                    <Badge variant="subtle">{t('workflowLibrary.nodeCount', { count: enrichment.nodeCount })}</Badge>
                  ) : null}
                  <WorkflowInputsBadge model={previewModel} />
                  {isCallable ? (
                    <Badge data-workflow-callable variant="subtle">
                      {t('workflowLibrary.callable')}
                    </Badge>
                  ) : null}
                  {missingNodePacks.length > 0 ? (
                    <Badge bg="bg.error" color="fg.error" data-workflow-missing-node-packs variant="subtle">
                      {t('workflowLibrary.nodePacksMissing', { count: missingNodePacks.length })}
                    </Badge>
                  ) : null}
                </HStack>
              ) : null}

              {item.description ? (
                <Text color="fg.muted" fontSize="xs" overflowWrap="anywhere">
                  {item.description}
                </Text>
              ) : null}

              {tags.length > 0 ? (
                <HStack flexWrap="wrap" gap="1" minW="0">
                  {tags.map((tag) => (
                    <Badge key={tag} variant="outline">
                      {tag}
                    </Badge>
                  ))}
                </HStack>
              ) : null}

              <WorkflowInputsSummary model={previewModel} onSeeAll={handlePreviewInputs} />
              <WorkflowAuthorNotes notes={workflowDocument?.notes ?? ''} />
              <WorkflowMissingNodePacks packs={missingNodePacks} />
              <WorkflowRequirementsList
                errorMessage={enrichment?.status === 'error' ? enrichment.message : null}
                resolved={resolved}
                onFindModel={handleFindModel}
              />
              {installableCount > 0 ? (
                <Button bg="bg.warning" color="fg.warning" size="sm" _hover={INSTALL_HOVER} onClick={handleInstall}>
                  <DownloadIcon />
                  {t('workflowLibrary.installModels', { count: installableCount })}
                </Button>
              ) : null}

              <DataList.Root data-workflow-detail-facts gap="1.5" orientation="horizontal">
                <WorkflowDetailFact label={t('workflowLibrary.factInProject')}>
                  {projectCopies.length === 0
                    ? t('workflowLibrary.projectCopiesNone')
                    : linkedRevision !== null
                      ? t('workflowLibrary.projectCopyRevision', { count: 1, revision: linkedRevision })
                      : t('workflowLibrary.projectCopies', { count: projectCopies.length })}
                </WorkflowDetailFact>
                {updatedWhen ? (
                  <WorkflowDetailFact label={t('workflowLibrary.factUpdated')}>
                    {t('workflowLibrary.whenAtRevision', { revision: item.revision, when: updatedWhen })}
                  </WorkflowDetailFact>
                ) : null}
                {createdWhen ? (
                  <WorkflowDetailFact label={t('workflowLibrary.factCreated')}>{createdWhen}</WorkflowDetailFact>
                ) : null}
                {showVisibility ? (
                  <WorkflowDetailFact label={t('workflowLibrary.visibility')}>
                    <HStack gap="2" justify="space-between" minW="0" w="full">
                      <Text
                        data-workflow-visibility={visibilityUnknown ? 'unknown' : itemIsPublic ? 'public' : 'private'}
                      >
                        {t(
                          visibilityUnknown
                            ? 'workflowLibrary.visibilityUnknown'
                            : itemIsPublic
                              ? 'workflowLibrary.public'
                              : 'workflowLibrary.private'
                        )}
                      </Text>
                      {canChangeVisibility && !visibilityUnknown ? (
                        <Switch.Root
                          checked={itemIsPublic === true}
                          colorPalette="accent"
                          disabled={isVisibilityPending || visibilityUnknown}
                          readOnly
                        >
                          <Switch.HiddenInput
                            key={itemIsPublic ? 'public' : 'private'}
                            onClick={handleVisibilityClick}
                            ref={visibilityInputRef}
                          />
                          <Switch.Control>
                            <Switch.Thumb />
                          </Switch.Control>
                          <Switch.Label srOnly>{t('workflowLibrary.publicWorkflow')}</Switch.Label>
                        </Switch.Root>
                      ) : null}
                    </HStack>
                  </WorkflowDetailFact>
                ) : null}
              </DataList.Root>
              {showVisibility ? (
                <Stack gap="1">
                  {canChangeVisibility && !visibilityUnknown ? (
                    <Text color="fg.muted" fontSize="xs">
                      {t('workflowLibrary.publicVisibilityHelp')}
                    </Text>
                  ) : null}
                  {visibilityError ? (
                    <Text color="fg.error" fontSize="xs" role="alert">
                      {t('workflowLibrary.visibilityUpdateFailed')}: {visibilityError}
                    </Text>
                  ) : null}
                  {visibilityUnknown ? (
                    <Button
                      size="xs"
                      disabled={visibilityState?.status === 'reconciling'}
                      onClick={handleRetryVisibilityRead}
                    >
                      {t('workflowLibrary.retry')}
                    </Button>
                  ) : null}
                </Stack>
              ) : null}
            </Stack>
          </Scrollable>

          {/* The pinned footer ends in the primary action; the narrow sheet's summary carries Open and More. */}
          <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" minW="0" p="2.5">
            <Button
              css={WORKFLOW_PREVIEW_TRIGGER_CSS}
              data-workflow-preview-trigger=""
              minW="0"
              size="lg"
              variant="outline"
              w="full"
              onClick={handlePreview}
            >
              <WorkflowIcon />
              {t('workflowLibrary.preview.action')}
            </Button>
            <HStack css={WORKFLOW_DETAIL_WIDE_ONLY_CSS} gap="2" minW="0">
              <Button data-workflow-detail-open flex="1" minW="0" size="lg" onClick={handleOpen}>
                {openLabel}
              </Button>
              <Menu.Root
                ids={moreActionsIds}
                positioning={menuPositioning}
                onRequestDismiss={keepMenuOpenWhenSiblingMenuCloses}
              >
                {/* The tooltip must sit inside the menu trigger in a dialog, or the menu never takes focus. */}
                <Menu.Trigger asChild>
                  <Tooltip content={t('workflowLibrary.moreActions')} ids={moreActionsIds}>
                    <IconButton aria-label={t('workflowLibrary.moreActions')} size="lg" variant="outline">
                      <EllipsisIcon />
                    </IconButton>
                  </Tooltip>
                </Menu.Trigger>
                <Portal>
                  <Menu.Positioner>
                    <MenuContent minW="min(13rem, var(--available-width))" maxW="var(--available-width)">
                      {actionItems}
                    </MenuContent>
                  </Menu.Positioner>
                </Portal>
              </Menu.Root>
            </HStack>
          </Stack>
        </WorkflowDetailSheet>
      ) : (
        <WorkflowDetailRailPlaceholder />
      )}
      {/* Outside the rail: a delete empties or retargets it while its confirmation animates out. */}
      {dialogs}
    </>
  );
});
