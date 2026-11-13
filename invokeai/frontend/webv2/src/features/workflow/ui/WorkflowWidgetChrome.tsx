import type { InvocationTemplate, ProjectWorkflowSource, XYPosition } from '@features/workflow/contracts';
import type * as ScrollableMenuModule from '@platform/ui/ScrollableMenuContent';

import { Box, HStack, Icon, Menu, Portal, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import { savedWorkflowDetailQueryOptions } from '@features/workflow/data/savedWorkflowQueries';
import { updateLoadedWorkflowNodes } from '@features/workflow/data/templates';
import { isBundledLibraryWorkflowId } from '@features/workflow/queries';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  buildConnectorNode,
  buildCurrentImageNode,
  buildInvocationNode,
  buildNotesNode,
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  createWorkflowId,
  getCompatibleInputTemplate,
  getCompatibleOutputTemplate,
  hasMultipleWorkflowReturnNodes,
  LOOP_LINKAGE_FIELD,
  resolveConnectorSource,
  shouldAddForReturnLoopLinkage,
  parseWorkflowJson,
  summarizeWorkflowLoadWarnings,
} from '@features/workflow/utility';
import { canCopyText } from '@platform/browser/clipboard';
import { createDeferredModule } from '@platform/react/deferredModule';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, IconButton, RenameDialog, Tooltip } from '@platform/ui';
import { MenuActionItem, MenuContent, SCROLLABLE_MENU_CONTENT_PROPS } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import {
  BookmarkIcon,
  BookUpIcon,
  CameraIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CloudAlertIcon,
  CloudCheckIcon,
  CloudIcon,
  CopyIcon,
  DownloadIcon,
  FileDownIcon,
  InfoIcon,
  LibraryIcon,
  Link2Icon,
  Link2OffIcon,
  LockIcon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
  TriangleAlertIcon,
  UploadIcon,
} from 'lucide-react';
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ElementType,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowWidgetLabelProps, WorkflowWidgetViewProps } from './contracts';
import type * as WorkflowLibraryDialogModule from './library/WorkflowLibraryDialog';
import type { WorkflowProjectPersistence, WorkflowProjectSnapshot } from './WorkflowUiContext';

import { AddNodeDialog } from './editor/AddNodeDialog';
import { CallSavedWorkflowSyncRuntime } from './editor/CallSavedWorkflowSyncRuntime';
import { getWorkflowViewCenter } from './editor/flowInstanceStore';
import { exportWorkflowPng, useWorkflowPngExportState } from './editor/workflowPngExport';
import { releaseWorkflowViewportsExcept } from './editor/workflowViewportStore';
import { LibraryCopyChoiceHost } from './library/LibraryCopyChoiceDialog';
import { isUpdatableSource } from './library/projectWorkflowEntries';
import { WorkflowPublicationHost } from './library/WorkflowPublicationHost';
import { PendingWorkflowLoader } from './PendingLibraryWorkflowLoader';
import { copyWorkflowJson, downloadWorkflowJson } from './workflowTransfer';
import {
  useWorkflowHostCommands,
  useWorkflowNotifications,
  useWorkflowPersistenceSelector,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from './WorkflowUiContext';
import {
  openWorkflowLibraryAtProjectWorkflow,
  requestWorkflowImport,
  requestWorkflowRename,
  requestWorkflowPublication,
  setAddNodeOpen,
  setWorkflowLibraryOpen,
  workflowUiStore,
} from './workflowUiStore';

/**
 * Keep dialogs mounted here and drive them through workflowUiStore; manifest slots contribute label, quick
 * actions, and shared menu actions.
 */

/**
 * The library dialog's chunk (cards, rails, toolbar, detail sheet) stays out of the editor's startup bundle until the
 * library first opens; its entry points preload it on hover or focus so that opening feels immediate. A failed load
 * (an offline moment, a redeploy) retries on the next attempt.
 */
let workflowLibraryDialogModule: Promise<typeof WorkflowLibraryDialogModule> | null = null;
const loadWorkflowLibraryDialog = (): Promise<typeof WorkflowLibraryDialogModule> =>
  (workflowLibraryDialogModule ??= import('./library/WorkflowLibraryDialog').catch((error: unknown) => {
    workflowLibraryDialogModule = null;
    throw error;
  }));
const preloadWorkflowLibraryDialog = (): void => {
  void loadWorkflowLibraryDialog().catch(() => {});
};
const LazyWorkflowLibraryDialog = lazy(() =>
  loadWorkflowLibraryDialog().then((module) => ({ default: module.WorkflowLibraryDialog }))
);

interface WorkflowChoice {
  id: string;
  name: string;
}

const selectWorkflowChoices = (project: WorkflowProjectSnapshot): WorkflowChoice[] =>
  project.workflows.map((entry) => ({ id: entry.document.id, name: entry.document.name }));
const areWorkflowChoicesEqual = (left: readonly WorkflowChoice[], right: readonly WorkflowChoice[]) =>
  left.length === right.length &&
  left.every((choice, index) => choice.id === right[index]?.id && choice.name === right[index].name);

/**
 * A long list opens with the active workflow in view. A frame later, once the positioner has capped the menu to the room
 * beside its trigger; the list's own viewport is the nearest scroller, so only the list moves.
 */
const revealActiveChoice = (item: HTMLElement | null) => {
  if (item) {
    requestAnimationFrame(() => item.scrollIntoView({ block: 'nearest' }));
  }
};

/**
 * The list's scroller is the shared scrolling menu list, which lives with the editor's boot code: imported here, it would
 * make this chunk, which a Launchpad manager reaches, reach the editor. It loads on first use instead, already loaded
 * wherever the editor has booted; until then, or if it fails to load, the same rows scroll natively. The rows and the
 * actions are this component's either way, so the menu opens whole and nothing moves under a click when it arrives.
 */
const scrollableMenu = createDeferredModule(() => import('@platform/ui/ScrollableMenuContent'));
const NATIVE_SCROLL_LIST_PROPS = { minH: '0', overflowY: 'auto' } as const;

/**
 * The project's workflows in collection order (the library's "Project order"), the active one checked, then a new
 * workflow and the way on to the library. Choosing one opens it as the library's Open does; the list scrolls, the two
 * actions under it stay.
 */
const WorkflowSwitcherContent = ({
  activeWorkflowId,
  onOpenLibrary,
}: {
  activeWorkflowId: string;
  onOpenLibrary: () => void;
}) => {
  const { t } = useTranslation();
  const choices = useWorkflowProjectSelector(selectWorkflowChoices, areWorkflowChoicesEqual);
  const { createWorkflow, selectWorkflow } = useProjectGraphCommands();
  // Nothing is replaced: a new workflow is added beside the others and becomes the active one, so nothing asks first.
  const newWorkflow = useCallback(() => createWorkflow(), [createWorkflow]);
  const chooseWorkflow = useCallback(
    (event: { value: string }) => {
      if (event.value !== activeWorkflowId) {
        selectWorkflow(event.value);
      }
    },
    [activeWorkflowId, selectWorkflow]
  );
  const footer = useMemo(
    () => (
      <>
        <Menu.Separator />
        <MenuActionItem
          icon={PlusIcon}
          label={t('widgets.workflow.newWorkflow')}
          value="new-workflow"
          onSelect={newWorkflow}
        />
        <MenuActionItem
          icon={LibraryIcon}
          label={t('widgets.workflow.libraryWithEllipsis')}
          value="workflow-library"
          onSelect={onOpenLibrary}
        />
      </>
    ),
    [newWorkflow, onOpenLibrary, t]
  );

  const rows = useMemo(
    () => (
      <Menu.RadioItemGroup value={activeWorkflowId} onValueChange={chooseWorkflow}>
        <Menu.ItemGroupLabel>{t('widgets.workflow.inThisProject')}</Menu.ItemGroupLabel>
        {choices.map((choice) => {
          const name = choice.name || t('widgets.workflow.untitled');

          return (
            // Typeahead matches an option's value text, which is its value (an id) unless named.
            <Menu.RadioItem
              key={choice.id}
              ref={choice.id === activeWorkflowId ? revealActiveChoice : undefined}
              value={choice.id}
              valueText={name}
            >
              <Menu.ItemIndicator color="accent.fg">
                <Icon as={CheckIcon} boxSize="3.5" />
              </Menu.ItemIndicator>
              <Menu.ItemText color={choice.name ? undefined : 'fg.subtle'} flex="1" minW="0">
                <MiddleTruncate as="span" text={name} />
              </Menu.ItemText>
            </Menu.RadioItem>
          );
        })}
      </Menu.RadioItemGroup>
    ),
    [activeWorkflowId, choices, chooseWorkflow, t]
  );
  const nativeList = useMemo(() => <Box {...NATIVE_SCROLL_LIST_PROPS}>{rows}</Box>, [rows]);
  const renderList = useCallback(
    (module: typeof ScrollableMenuModule) => <module.ScrollableMenuList>{rows}</module.ScrollableMenuList>,
    [rows]
  );

  return (
    <MenuContent {...SCROLLABLE_MENU_CONTENT_PROPS} maxW="20rem" minW="14rem">
      <scrollableMenu.Deferred fallback={nativeList} render={renderList} />
      {footer}
    </MenuContent>
  );
};

/** How the project's own persistence is shown beside the library actions. */
const PERSISTENCE_VIEW: Record<
  WorkflowProjectPersistence['status'],
  { color: string; icon: ElementType; labelKey: string }
> = {
  conflict: { color: 'fg.warning', icon: TriangleAlertIcon, labelKey: 'widgets.workflow.persistenceConflict' },
  error: { color: 'fg.error', icon: CloudAlertIcon, labelKey: 'widgets.workflow.persistenceError' },
  pending: { color: 'fg.subtle', icon: CloudIcon, labelKey: 'widgets.workflow.persistencePending' },
  saved: { color: 'fg.subtle', icon: CloudCheckIcon, labelKey: 'widgets.workflow.persistenceSaved' },
  saving: { color: 'fg.subtle', icon: RefreshCwIcon, labelKey: 'widgets.workflow.persistenceSaving' },
};

/**
 * What the workflow's library link says: the template it updates, a bundled template, another linked template, or
 * none. The template's name is not part of the project; it is read only once asked for, through the query the update
 * confirmation uses.
 */
const useTemplateLinkDescription = (
  source: ProjectWorkflowSource | undefined,
  wantsName: boolean,
  session: ReturnType<typeof useAuthSession>
): string => {
  const { t } = useTranslation();
  const isLinked = isUpdatableSource(source, session);
  const templateName = useQuery({
    ...savedWorkflowDetailQueryOptions(isLinked && source ? source.libraryWorkflowId : ''),
    enabled: isLinked && wantsName,
    select: (record) => record.name,
  }).data;

  return isLinked
    ? templateName
      ? t('widgets.workflow.linkedTemplate', { name: templateName })
      : t('widgets.workflow.linkedTemplateUnnamed')
    : source
      ? isBundledLibraryWorkflowId(source.libraryWorkflowId)
        ? t('workflowLibrary.sourceBundled')
        : source.revision === null
          ? t('workflowLibrary.sourceUnknownRevision')
          : t('workflowLibrary.sourceRevision', { revision: source.revision })
      : t('workflowLibrary.sourceNone');
};

/**
 * The center chrome's label: the workflow's name, with its library link inside the same trigger, so the name pill
 * shrinks as one piece. It opens a menu that switches between the project's workflows or opens the library. The label
 * clips rather than overflows its island: at narrow center widths the actions island beside it would otherwise paint
 * over the name and take its clicks.
 */
export const WorkflowWidgetLabel = ({ region }: WorkflowWidgetLabelProps) => {
  const { t } = useTranslation();
  const workflowName = useWorkflowProjectSelector((project) => project.projectGraph.name);
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const source = useWorkflowProjectSelector((project) => project.activeWorkflow.source);
  const session = useAuthSession();
  const triggerIds = useTooltipTriggerIds();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [wantsTemplateName, setWantsTemplateName] = useState(false);
  const linkDescription = useTemplateLinkDescription(source, wantsTemplateName, session);
  const linkDescriptionId = useId();
  const openProjectWorkflows = useCallback(
    () => openWorkflowLibraryAtProjectWorkflow(projectId, activeWorkflowId),
    [activeWorkflowId, projectId]
  );
  const onMenuOpenChange = useCallback((event: { open: boolean }) => setIsMenuOpen(event.open), []);
  const onTooltipOpenChange = useCallback((event: { open: boolean }) => {
    if (event.open) {
      setWantsTemplateName(true);
    }
  }, []);

  const tooltipContent = useMemo(
    () => (
      <Stack gap="0.5">
        <Text>{t('widgets.workflow.switchWorkflow')}</Text>
        <Text color="fg.muted">{linkDescription}</Text>
      </Stack>
    ),
    [linkDescription, t]
  );

  if (region !== 'center') {
    return <Text fontWeight="700">{t('widgets.labels.workflow')}</Text>;
  }

  // Center chrome already names the widget; the workflow name switches between the project's workflows.
  const displayName = workflowName || t('widgets.workflow.untitled');
  const isLinked = isUpdatableSource(source, session);
  // A source this account cannot update is locked, whether bundled or owned by another account.
  const linkIcon = isLinked ? Link2Icon : source ? LockIcon : Link2OffIcon;
  const linkKind = isLinked
    ? 'linked'
    : source
      ? isBundledLibraryWorkflowId(source.libraryWorkflowId)
        ? 'bundled'
        : 'read-only'
      : 'none';

  return (
    // Sized by its content, so a narrow center shrinks it and the view menu beside it in proportion; a zero basis
    // would leave the name only what the view menu does not take.
    <HStack css={LABEL_CLIP_CSS} flex="1 1 auto" gap="1" minW="0">
      <Text color="fg.subtle" flexShrink={1} minW="0" overflow="hidden">
        /
      </Text>
      {/* The list mounts only while open: the chrome re-renders on editor changes the list does not show. */}
      <Menu.Root
        ids={triggerIds}
        lazyMount
        positioning={WORKFLOW_SWITCHER_POSITIONING}
        unmountOnExit
        onOpenChange={onMenuOpenChange}
      >
        {/* Not over the menu it describes: it opens below the trigger, where the tooltip would cover the list. */}
        <Tooltip
          content={tooltipContent}
          disabled={isMenuOpen}
          ids={triggerIds}
          keepsRootWhenDisabled
          onOpenChange={onTooltipOpenChange}
        >
          <Menu.Trigger asChild>
            <Button
              aria-describedby={linkDescriptionId}
              aria-label={t('widgets.workflow.switchWorkflowTrigger', { name: displayName })}
              // The button recipe refuses to shrink; the name must give way before the actions island overlaps it.
              flexShrink={1}
              maxW="16rem"
              minW="0"
              overflow="hidden"
              size="sm"
              variant="ghost"
              onFocus={preloadWorkflowLibraryDialog}
              onPointerEnter={preloadWorkflowLibraryDialog}
            >
              <MiddleTruncate
                color={workflowName ? undefined : 'fg.subtle'}
                fontWeight="600"
                minW="0"
                text={displayName}
              />
              <Icon as={linkIcon} boxSize="3.5" color="fg.subtle" data-template-link={linkKind} flexShrink={0} />
              <Icon as={ChevronDownIcon} boxSize="3" flexShrink={0} />
            </Button>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <WorkflowSwitcherContent activeWorkflowId={activeWorkflowId} onOpenLibrary={openProjectWorkflows} />
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <VisuallyHidden id={linkDescriptionId}>{linkDescription}</VisuallyHidden>
    </HStack>
  );
};

const WORKFLOW_SWITCHER_POSITIONING = { placement: 'bottom-start' } as const;

// Clipped, but with room for the name button's focus ring.
const LABEL_CLIP_CSS = { overflow: 'clip', overflowClipMargin: '4px' } as const;

const EXPORT_SUBMENU_POSITIONING = { placement: 'right-start' } as const;

/** Entries contributed to the shared widget actions menu, under its graph actions. */
export const WorkflowMenuItems = (_props: WorkflowWidgetViewProps) => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const { getProjectGraph } = useWorkflowUi();
  const { widgets } = useWorkflowHostCommands();
  const notify = useWorkflowNotifications();
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const source = useWorkflowProjectSelector((project) => project.activeWorkflow.source);
  const workflowName = useWorkflowProjectSelector((project) => project.activeWorkflow.document.name);
  const pngExport = useWorkflowPngExportState();
  // One export runs at a time, from here or the editor toolbar.
  const downloadPngUnavailableReason = pngExport.isExporting
    ? t('widgets.workflow.downloadPngRunning')
    : pngExport.isAvailable
      ? undefined
      : t('widgets.workflow.downloadPngUnavailable');
  const canUpdateSource = isUpdatableSource(source, session);

  const openDetailsPanel = useCallback(() => {
    widgets.open({ region: 'left', widgetId: 'workflow' });
    widgets.patchValues('workflow', { editTab: 'details', panelMode: 'edit' });
  }, [widgets]);
  const exportWorkflow = useCallback(() => {
    const projectGraph = getProjectGraph();

    if (hasMultipleWorkflowReturnNodes(projectGraph)) {
      notify.error(t('projects.exportFailed'), t('workflowLibrary.multipleWorkflowReturnNodesForTransfer'));
      return;
    }

    downloadWorkflowJson(projectGraph);
  }, [getProjectGraph, notify, t]);
  const copyUnavailableReason = canCopyText() ? undefined : t('common.copyUnavailable');
  const copyWorkflow = useCallback(() => {
    const projectGraph = getProjectGraph();

    if (hasMultipleWorkflowReturnNodes(projectGraph)) {
      notify.error(t('widgets.workflow.copyJsonFailed'), t('workflowLibrary.multipleWorkflowReturnNodesForTransfer'));
      return;
    }

    copyWorkflowJson(projectGraph)
      .then(() => notify.success(t('widgets.workflow.copyJsonSuccess')))
      .catch(() => notify.error(t('widgets.workflow.copyJsonFailed')));
  }, [getProjectGraph, notify, t]);
  const downloadPng = useCallback(
    () => void exportWorkflowPng({ notifications: notify, t, workflowName }),
    [notify, t, workflowName]
  );
  const saveToLibrary = useCallback(
    () => requestWorkflowPublication({ kind: 'save-as-new', workflowId: activeWorkflowId }),
    [activeWorkflowId]
  );
  const updateTemplate = useCallback(
    () => requestWorkflowPublication({ kind: 'update-source', workflowId: activeWorkflowId }),
    [activeWorkflowId]
  );
  const renameActiveWorkflow = useCallback(() => requestWorkflowRename(activeWorkflowId), [activeWorkflowId]);

  return (
    <>
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('widgets.workflow.menuThisWorkflow')}</Menu.ItemGroupLabel>
        <MenuActionItem
          icon={PencilIcon}
          label={t('widgets.workflow.renameWithEllipsis')}
          value="rename"
          onSelect={renameActiveWorkflow}
        />
        <MenuActionItem
          icon={InfoIcon}
          label={t('widgets.workflow.detailsWithEllipsis')}
          value="details"
          onSelect={openDetailsPanel}
        />
        {/* Saving always creates a template. Beside Update, a linked workflow's save says it makes a new one. */}
        <MenuActionItem
          icon={BookmarkIcon}
          label={t(
            canUpdateSource
              ? 'widgets.workflow.saveAsNewTemplateWithEllipsis'
              : 'widgets.workflow.saveToLibraryWithEllipsis'
          )}
          value="save-to-library"
          onSelect={saveToLibrary}
        />
        {canUpdateSource ? (
          <MenuActionItem
            icon={BookUpIcon}
            label={t('widgets.workflow.updateTemplateWithEllipsis')}
            value="update-template"
            onSelect={updateTemplate}
          />
        ) : null}
      </Menu.ItemGroup>
      <Menu.Separator />
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel>{t('widgets.workflow.menuImportExport')}</Menu.ItemGroupLabel>
        <MenuActionItem
          icon={UploadIcon}
          label={t('widgets.workflow.importJsonWithEllipsis')}
          value="import"
          onSelect={requestWorkflowImport}
        />
        <Menu.Root positioning={EXPORT_SUBMENU_POSITIONING}>
          <Menu.TriggerItem>
            <HStack gap="2" minW="0" w="full">
              <Icon as={FileDownIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
              <Text flex="1" fontSize="md">
                {t('widgets.workflow.export')}
              </Text>
              <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
            </HStack>
          </Menu.TriggerItem>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="12rem">
                <MenuActionItem
                  icon={DownloadIcon}
                  label={t('widgets.workflow.exportJson')}
                  value="export-json"
                  onSelect={exportWorkflow}
                />
                <MenuActionItem
                  icon={CopyIcon}
                  label={t('widgets.workflow.copyJson')}
                  unavailableReason={copyUnavailableReason}
                  value="copy-json"
                  onSelect={copyWorkflow}
                />
                <MenuActionItem
                  icon={CameraIcon}
                  label={t('widgets.workflow.downloadPng')}
                  unavailableReason={downloadPngUnavailableReason}
                  value="download-png"
                  onSelect={downloadPng}
                />
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Menu.ItemGroup>
    </>
  );
};

const selectPersistenceView = (persistence: WorkflowProjectPersistence) => ({
  hasLocalRecovery: persistence.hasLocalRecovery,
  status: persistence.status,
});

export const WorkflowHeaderActions = ({ region }: WorkflowWidgetViewProps) => {
  const { t } = useTranslation();
  const session = useAuthSession();
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const canUpdateSource = useWorkflowProjectSelector((project) =>
    isUpdatableSource(project.activeWorkflow.source, session)
  );
  const { hasLocalRecovery, status } = useWorkflowPersistenceSelector(selectPersistenceView);
  const openAddNode = useCallback(() => setAddNodeOpen(true), []);
  const openWorkflowLibrary = useCallback(() => setWorkflowLibraryOpen(true), []);
  const saveToLibrary = useCallback(
    () => requestWorkflowPublication({ kind: 'save-as-new', workflowId: activeWorkflowId }),
    [activeWorkflowId]
  );
  const updateTemplate = useCallback(
    () => requestWorkflowPublication({ kind: 'update-source', workflowId: activeWorkflowId }),
    [activeWorkflowId]
  );
  const view = PERSISTENCE_VIEW[status];
  // A pending save without browser recovery has no safety net at all; say so rather than showing the same cloud.
  const persistenceLabel =
    status === 'pending' && !hasLocalRecovery ? t('widgets.workflow.persistencePendingNoRecovery') : t(view.labelKey);

  return (
    <HStack gap="0.5">
      {region === 'center' ? (
        <Tooltip content={t('widgets.workflow.addNode')}>
          <IconButton
            aria-label={t('widgets.workflow.addNode')}
            color="fg.muted"
            size="sm"
            variant="ghost"
            onClick={openAddNode}
          >
            <Icon as={PlusIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
      {/* The center label's menu already opens the library; other regions need this separate trigger and leave
          publication and persistence to the editor header. */}
      {region === 'center' ? null : (
        <Tooltip content={t('widgets.workflow.library')}>
          <IconButton
            aria-label={t('widgets.workflow.library')}
            color="fg.muted"
            size="sm"
            variant="ghost"
            onClick={openWorkflowLibrary}
            onFocus={preloadWorkflowLibraryDialog}
            onPointerEnter={preloadWorkflowLibraryDialog}
          >
            <Icon as={LibraryIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      )}
      {region === 'center' ? (
        <>
          <Tooltip content={t('widgets.workflow.saveToLibraryWithEllipsis')}>
            <IconButton
              aria-label={t('widgets.workflow.saveToLibraryWithEllipsis')}
              color="fg.muted"
              size="sm"
              variant="ghost"
              onClick={saveToLibrary}
            >
              <Icon as={BookmarkIcon} boxSize="3.5" />
            </IconButton>
          </Tooltip>
          {canUpdateSource ? (
            <Tooltip content={t('widgets.workflow.updateTemplateWithEllipsis')}>
              <IconButton
                aria-label={t('widgets.workflow.updateTemplateWithEllipsis')}
                color="fg.muted"
                size="sm"
                variant="ghost"
                onClick={updateTemplate}
              >
                <Icon as={BookUpIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
          <Tooltip content={persistenceLabel}>
            <Box
              alignItems="center"
              aria-label={persistenceLabel}
              boxSize="6"
              color={view.color}
              data-persistence-status={status}
              display="flex"
              justifyContent="center"
              role="status"
            >
              <Icon as={view.icon} boxSize="3.5" />
            </Box>
          </Tooltip>
        </>
      ) : null}
    </HStack>
  );
};

export const WorkflowDialogHost = () => {
  const { addWorkflow, editGraph, renameWorkflow } = useProjectGraphCommands();
  const { project: projectStore } = useWorkflowUi();
  const notify = useWorkflowNotifications();
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const addNodeConnection = workflowUiStore.useSelector((snapshot) => snapshot.addNodeConnection);
  const addNodePosition = workflowUiStore.useSelector((snapshot) => snapshot.addNodePosition);
  const importRequestCount = workflowUiStore.useSelector((snapshot) => snapshot.importRequestCount);
  const isAddNodeOpen = workflowUiStore.useSelector((snapshot) => snapshot.isAddNodeOpen);
  const isLibraryOpen = workflowUiStore.useSelector((snapshot) => snapshot.isLibraryOpen);
  const [hasOpenedLibrary, setHasOpenedLibrary] = useState(isLibraryOpen);
  if (isLibraryOpen && !hasOpenedLibrary) {
    setHasOpenedLibrary(true);
  }
  const lastImportRequestRef = useRef(importRequestCount);
  // A rename request opens the dialog once, for the workflow it named, and only while that workflow is the active
  // one; closing records the request as handled.
  const renameRequest = workflowUiStore.useSelector((snapshot) => snapshot.renameRequest);
  const [handledRenameRequestId, setHandledRenameRequestId] = useState(0);
  const activeWorkflowId = useWorkflowProjectSelector((project) => project.activeWorkflowId);
  const renameTargetId = renameRequest?.workflowId ?? null;
  const renameTargetName = useWorkflowProjectSelector(
    (project) => project.workflows.find((entry) => entry.document.id === renameTargetId)?.document.name ?? ''
  );
  const isRenameOpen =
    renameRequest !== null &&
    renameRequest.requestId > handledRenameRequestId &&
    renameRequest.workflowId === activeWorkflowId;

  // A request whose workflow stopped being active is over; it must not come back when that workflow returns.
  if (renameRequest !== null && renameRequest.requestId > handledRenameRequestId && !isRenameOpen) {
    setHandledRenameRequestId(renameRequest.requestId);
  }
  const closeRename = useCallback(
    () => setHandledRenameRequestId(renameRequest?.requestId ?? 0),
    [renameRequest?.requestId]
  );
  const submitRename = useCallback(
    (name: string) => {
      if (renameTargetId !== null) {
        renameWorkflow(renameTargetId, name);
      }
    },
    [renameTargetId, renameWorkflow]
  );

  // Editor session state (viewports) is keyed by workflow; release it once its workflow leaves the project.
  useMountEffect(() => {
    let lastMembership = '';

    return projectStore.subscribe(() => {
      const snapshot = projectStore.getSnapshot();
      const liveWorkflowIds = snapshot.workflows.map((entry) => entry.document.id);
      const membership = `${snapshot.id}\0${liveWorkflowIds.join('\0')}`;

      if (membership === lastMembership) {
        return;
      }

      lastMembership = membership;
      releaseWorkflowViewportsExcept(snapshot.id, liveWorkflowIds);
    });
  });

  useEffect(() => {
    if (importRequestCount > lastImportRequestRef.current) {
      fileInputRef.current?.click();
    }

    lastImportRequestRef.current = importRequestCount;
  }, [importRequestCount]);

  const getInsertPosition = useCallback((): XYPosition => {
    if (addNodePosition) {
      return addNodePosition;
    }

    // The middle of the editor, wherever its window or panel sits, not of the browser window.
    const center = getWorkflowViewCenter() ?? { x: 0, y: 0 };

    // Slight scatter so repeated inserts do not stack perfectly.
    return { x: center.x + (Math.random() - 0.5) * 80, y: center.y + (Math.random() - 0.5) * 80 };
  }, [addNodePosition]);

  const addNode = useCallback(
    (template: InvocationTemplate) => {
      const node = buildInvocationNode(template, getInsertPosition());

      if (!addNodeConnection) {
        editGraph({ node, type: 'addNode' });
        return;
      }

      if (addNodeConnection.kind === 'source') {
        const targetInput =
          addNodeConnection.sourceHandle === LOOP_LINKAGE_FIELD && template.type === 'for_return'
            ? template.inputs[LOOP_LINKAGE_FIELD]
            : template.type === 'for_return'
              ? template.inputs.output
              : getCompatibleInputTemplate(template, addNodeConnection.sourceType);

        if (!targetInput) {
          editGraph({ node, type: 'addNode' });
          return;
        }

        const edge = {
          id: createWorkflowId('edge'),
          source: addNodeConnection.sourceNodeId,
          sourceHandle: addNodeConnection.sourceHandle,
          target: node.id,
          targetHandle: targetInput.name,
          type:
            addNodeConnection.sourceHandle === LOOP_LINKAGE_FIELD && template.type === 'for_return'
              ? ('loop_linkage' as const)
              : ('default' as const),
        };

        const currentGraph = projectStore.getSnapshot().projectGraph;
        const sourceNode = currentGraph.nodes.find((candidate) => candidate.id === addNodeConnection.sourceNodeId);
        const resolvedSource =
          sourceNode?.type === 'connector'
            ? resolveConnectorSource(sourceNode.id, currentGraph.nodes, currentGraph.edges)
            : sourceNode?.type === 'invocation'
              ? { fieldName: addNodeConnection.sourceHandle, nodeId: sourceNode.id, type: null }
              : null;
        const resolvedSourceNode = currentGraph.nodes.find((candidate) => candidate.id === resolvedSource?.nodeId);
        const shouldAddLoopLinkage = shouldAddForReturnLoopLinkage(
          template.type,
          resolvedSource,
          resolvedSourceNode,
          currentGraph.edges
        );

        const edges = [edge];
        if (shouldAddLoopLinkage && resolvedSourceNode?.type === 'invocation') {
          edges.push({
            id: createWorkflowId('edge'),
            source: resolvedSourceNode.id,
            sourceHandle: LOOP_LINKAGE_FIELD,
            target: node.id,
            targetHandle: LOOP_LINKAGE_FIELD,
            type: 'loop_linkage',
          });
        }

        editGraph({ edge: edges, node, type: 'addNodeAndEdge' });
        return;
      }

      const sourceOutput =
        addNodeConnection.targetHandle === LOOP_LINKAGE_FIELD && template.type === 'for'
          ? template.outputs[LOOP_LINKAGE_FIELD]
          : getCompatibleOutputTemplate(template, addNodeConnection.targetType);

      if (!sourceOutput) {
        editGraph({ node, type: 'addNode' });
        return;
      }

      editGraph({
        edge: {
          id: createWorkflowId('edge'),
          source: node.id,
          sourceHandle: sourceOutput.name,
          target: addNodeConnection.targetNodeId,
          targetHandle: addNodeConnection.targetHandle,
          type:
            addNodeConnection.targetHandle === LOOP_LINKAGE_FIELD && template.type === 'for'
              ? 'loop_linkage'
              : 'default',
        },
        node,
        type: 'addNodeAndEdge',
      });
    },
    [addNodeConnection, editGraph, getInsertPosition, projectStore]
  );

  const addNote = useCallback(() => {
    editGraph({ node: buildNotesNode(getInsertPosition()), type: 'addNode' });
  }, [editGraph, getInsertPosition]);

  const addConnector = useCallback(() => {
    const node = buildConnectorNode(getInsertPosition());

    if (!addNodeConnection) {
      editGraph({ node, type: 'addNode' });
      return;
    }

    editGraph({
      edge:
        addNodeConnection.kind === 'source'
          ? {
              id: createWorkflowId('edge'),
              source: addNodeConnection.sourceNodeId,
              sourceHandle: addNodeConnection.sourceHandle,
              target: node.id,
              targetHandle: CONNECTOR_INPUT_HANDLE,
              type: 'default',
            }
          : {
              id: createWorkflowId('edge'),
              source: node.id,
              sourceHandle: CONNECTOR_OUTPUT_HANDLE,
              target: addNodeConnection.targetNodeId,
              targetHandle: addNodeConnection.targetHandle,
              type: 'default',
            },
      node,
      type: 'addNodeAndEdge',
    });
  }, [addNodeConnection, editGraph, getInsertPosition]);

  const addCurrentImage = useCallback(() => {
    editGraph({ node: buildCurrentImageNode(getInsertPosition()), type: 'addNode' });
  }, [editGraph, getInsertPosition]);

  // A file is portable: whatever id it carries names nothing this project may write to.
  const importFile = useCallback(
    (file: File) => {
      file
        .text()
        .then((text) => {
          const { document: parsed, warnings: parseWarnings } = parseWorkflowJson(JSON.parse(text));
          const { document, warnings: updateWarnings } = updateLoadedWorkflowNodes(parsed, t);
          const warnings = [...parseWarnings, ...updateWarnings];

          addWorkflow(document, {
            label: t('widgets.workflow.importedLabel', { name: file.name }),
            reusePlaceholder: true,
          });

          const warningSummary = summarizeWorkflowLoadWarnings(warnings);

          if (warningSummary) {
            notify.info(t('widgets.workflow.importWarning'), warningSummary);
          }
        })
        .catch((error: unknown) => {
          notify.error(
            t('widgets.workflow.importFailed'),
            error instanceof Error ? error.message : t('widgets.workflow.importInvalid')
          );
        });
    },
    [addWorkflow, notify, t]
  );
  const handleImportFile = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      event.currentTarget.value = '';

      if (file) {
        importFile(file);
      }
    },
    [importFile]
  );

  return (
    <>
      <CallSavedWorkflowSyncRuntime />
      <input ref={fileInputRef} accept=".json,application/json" hidden type="file" onChange={handleImportFile} />
      <AddNodeDialog
        connectionFilter={addNodeConnection}
        isOpen={isAddNodeOpen}
        onAddCurrentImage={addCurrentImage}
        onAddConnector={addConnector}
        onAddNode={addNode}
        onAddNote={addNote}
        onOpenChange={setAddNodeOpen}
      />
      {/* Loaded on the first opening and kept mounted after, so its exit transitions, reset on close and remembered
          place behave as for an always-mounted dialog. Its own busy overlay covers loading once it shows. */}
      {hasOpenedLibrary ? (
        <Suspense fallback={null}>
          <LazyWorkflowLibraryDialog isOpen={isLibraryOpen} onOpenChange={setWorkflowLibraryOpen} />
        </Suspense>
      ) : null}
      <LibraryCopyChoiceHost />
      <RenameDialog
        key={renameRequest?.requestId ?? 0}
        initialName={renameTargetName}
        isOpen={isRenameOpen}
        label={t('workflowLibrary.workflowName')}
        submitLabel={t('workflowLibrary.rename')}
        title={t('workflowLibrary.renameTitle')}
        onClose={closeRename}
        onSubmit={submitRename}
      />
      <WorkflowPublicationHost />
      <PendingWorkflowLoader />
    </>
  );
};
