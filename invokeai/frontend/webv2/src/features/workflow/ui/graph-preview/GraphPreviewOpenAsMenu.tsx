import type { WorkflowInvocationSourceId, WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
import type { ReactNode } from 'react';

import { Menu, Portal } from '@chakra-ui/react';
import { previewGraphToDocument } from '@features/workflow/core/graphToDocument';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useWorkflowPublication } from '@features/workflow/ui/library/useWorkflowPublication';
import { MenuActionItem } from '@features/workflow/ui/MenuActionItem';
import {
  useWorkflowGraphPreview,
  useWorkflowHostCommands,
  useWorkflowNotifications,
} from '@features/workflow/ui/WorkflowUiContext';
import { downloadText } from '@platform/browser/downloadBlob';
import { MenuContent } from '@platform/ui/Menu';
import { BookmarkIcon, DownloadIcon, GitForkIcon, PencilRulerIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const MENU_POSITIONING = { placement: 'top-end' } as const;

interface GraphPreviewOpenAsMenuProps {
  children: ReactNode;
  graph: WorkflowPreviewGraph;
  sourceId?: WorkflowInvocationSourceId;
  sourceLabel: string;
  /** Dialog close after a successful "edit in editor". */
  onClose: () => void;
}

/**
 * Convert previews into editor/library/project documents or download raw JSON. Keep the trigger in the dialog to
 * avoid another platform-barrel dependency.
 */
export const GraphPreviewOpenAsMenu = ({
  children,
  graph,
  sourceId,
  sourceLabel,
  onClose,
}: GraphPreviewOpenAsMenuProps) => {
  const { t } = useTranslation();
  const { workflows } = useWorkflowHostCommands();
  const graphPreview = useWorkflowGraphPreview();
  const notifications = useWorkflowNotifications();
  const { saveDocumentAsNew } = useWorkflowPublication();
  // A hook (not `getInvocationTemplatesSnapshot()`) so the "Edit in workflow
  // editor" item's disabled state updates live if the menu is opened while
  // templates are still loading.
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const canEditInEditor = sourceId !== 'workflow';
  const isTemplatesLoaded = templatesSnapshot.status === 'loaded';

  const handleEditInEditor = useCallback(() => {
    const { document, skippedNodeTypes } = previewGraphToDocument(graph, templatesSnapshot.templates);

    if (document.nodes.length === 0) {
      notifications.error(t('graphPreview.editInEditorFailed'));
      return;
    }

    if (skippedNodeTypes.length > 0) {
      notifications.info(
        t('graphPreview.nodesSkipped', { count: skippedNodeTypes.length, types: skippedNodeTypes.join(', ') })
      );
    }

    // The preview becomes a workflow of its own beside the project's others (or takes over an untouched blank).
    workflows.addWorkflow(document, { label: t('graphPreview.openedFromPreview'), reusePlaceholder: true });
    graphPreview.openWorkflowEditor();
    onClose();
  }, [graph, templatesSnapshot, notifications, t, workflows, graphPreview, onClose]);

  const saveToLibrary = useCallback(async () => {
    const { document, skippedNodeTypes } = previewGraphToDocument(graph, templatesSnapshot.templates);

    if (document.nodes.length === 0) {
      notifications.error(t('graphPreview.saveToLibraryFailed'));
      return;
    }

    if (skippedNodeTypes.length > 0) {
      notifications.info(
        t('graphPreview.nodesSkipped', { count: skippedNodeTypes.length, types: skippedNodeTypes.join(', ') })
      );
    }

    document.name = graph.label ?? sourceLabel;

    // The publication reports its own outcome; a lost answer is retried by the same captured request.
    const result = await saveDocumentAsNew(document, document.name);

    if (result.status === 'failed') {
      const retried = await result.retry();

      if (retried.status === 'failed') {
        notifications.error(t('graphPreview.saveToLibraryFailed'), retried.message);
      }
    }
  }, [graph, templatesSnapshot, sourceLabel, saveDocumentAsNew, notifications, t]);
  const handleSaveToLibrary = useCallback(() => void saveToLibrary(), [saveToLibrary]);

  const handleForkIntoProject = useCallback(() => {
    const { document, skippedNodeTypes } = previewGraphToDocument(graph, templatesSnapshot.templates);

    if (document.nodes.length === 0) {
      notifications.error(t('graphPreview.forkIntoProjectFailed'));
      return;
    }

    if (skippedNodeTypes.length > 0) {
      notifications.info(
        t('graphPreview.nodesSkipped', { count: skippedNodeTypes.length, types: skippedNodeTypes.join(', ') })
      );
    }

    document.name = graph.label ?? sourceLabel;

    // The port creates + activates a fresh project before loading the
    // document, so the project this preview came from is left untouched.
    graphPreview.openDocumentInNewProject(document, t('graphPreview.openedFromPreview'));
    onClose();
  }, [graph, templatesSnapshot, sourceLabel, notifications, t, graphPreview, onClose]);

  const handleDownloadJson = useCallback(() => {
    const fileName = `${(graph.label ?? 'graph').trim().replaceAll(/\s+/g, '-').toLowerCase()}.json`;
    downloadText(JSON.stringify(graph.backendGraph ?? graph, null, 2), fileName, 'application/json');
  }, [graph]);

  return (
    <Menu.Root positioning={MENU_POSITIONING}>
      <Menu.Trigger asChild>{children}</Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="16rem">
            {canEditInEditor ? (
              <MenuActionItem
                hint={t('graphPreview.editInEditorHint')}
                icon={PencilRulerIcon}
                isDisabled={!isTemplatesLoaded}
                label={t('graphPreview.editInEditor')}
                value="edit-in-editor"
                onSelect={handleEditInEditor}
              />
            ) : null}
            <MenuActionItem
              hint={t('graphPreview.saveToLibraryHint')}
              icon={BookmarkIcon}
              isDisabled={!isTemplatesLoaded}
              label={t('graphPreview.saveToLibrary')}
              value="save-to-library"
              onSelect={handleSaveToLibrary}
            />
            <MenuActionItem
              hint={t('graphPreview.forkIntoProjectHint')}
              icon={GitForkIcon}
              isDisabled={!isTemplatesLoaded}
              label={t('graphPreview.forkIntoProject')}
              value="fork-into-project"
              onSelect={handleForkIntoProject}
            />
            <MenuActionItem
              hint={t('graphPreview.downloadJsonHint')}
              icon={DownloadIcon}
              label={t('graphPreview.downloadJson')}
              value="download-json"
              onSelect={handleDownloadJson}
            />
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
