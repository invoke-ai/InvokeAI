import type { WorkflowInvocationSourceId, WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
import type { ReactNode } from 'react';

import { Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { previewGraphToDocument } from '@features/workflow/core/graphToDocument';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useSaveWorkflowToLibrary } from '@features/workflow/ui/library/useSaveWorkflowToLibrary';
import {
  useWorkflowGraphPreview,
  useWorkflowHostCommands,
  useWorkflowNotifications,
} from '@features/workflow/ui/WorkflowUiContext';
import { downloadText } from '@platform/browser/downloadBlob';
import { BookmarkIcon, DownloadIcon, GitForkIcon, PencilRulerIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const MENU_POSITIONING = { placement: 'top-end' } as const;
const DISABLED_PROPS = { opacity: 0.4 };
// Items carry a two-line body (label + hint), so the icon top-aligns with the
// label line instead of floating centered against both.
const MENU_ITEM_LAYOUT = { alignItems: 'flex-start', gap: '2.5', py: '1.5' } as const;
const MENU_ITEM_ICON_PROPS = { boxSize: '3.5', flexShrink: 0, mt: '0.5' } as const;

interface GraphPreviewOpenAsMenuProps {
  children: ReactNode;
  graph: WorkflowPreviewGraph;
  sourceId?: WorkflowInvocationSourceId;
  sourceLabel: string;
  /** Dialog close after a successful "edit in editor". */
  onClose: () => void;
}

const MenuItemBody = ({ hint, label }: { hint: string; label: string }) => (
  <Stack gap="0" minW="0">
    <Menu.ItemText>{label}</Menu.ItemText>
    <Text color="fg.subtle" fontSize="2xs">
      {hint}
    </Text>
  </Stack>
);

/**
 * The graph preview dialog's "Open as" menu — four ways to hand a
 * read-only preview graph off to something that keeps it: convert it into an
 * editable document and load it into the workflow editor, save it to the
 * workflow library without touching the active project, fork it into a fresh
 * project, or download the raw JSON. The trigger button lives in the dialog (it needs `@platform/ui`'s
 * `Button`); this file stays out of that import so it doesn't add to that
 * package's fan-in.
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
  const { saveDocumentAsNew } = useSaveWorkflowToLibrary();
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

    // `replace` emits its own success notification and undo entry.
    workflows.replace(document, t('graphPreview.openedFromPreview'));
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

    const id = await saveDocumentAsNew(document);

    if (id !== null) {
      notifications.success(t('graphPreview.savedToLibrary'));
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
          <Menu.Content minW="16rem">
            {canEditInEditor ? (
              <Menu.Item
                {...MENU_ITEM_LAYOUT}
                _disabled={DISABLED_PROPS}
                disabled={!isTemplatesLoaded}
                value="edit-in-editor"
                onClick={handleEditInEditor}
              >
                <Icon as={PencilRulerIcon} {...MENU_ITEM_ICON_PROPS} />
                <MenuItemBody hint={t('graphPreview.editInEditorHint')} label={t('graphPreview.editInEditor')} />
              </Menu.Item>
            ) : null}
            <Menu.Item
              {...MENU_ITEM_LAYOUT}
              _disabled={DISABLED_PROPS}
              disabled={!isTemplatesLoaded}
              value="save-to-library"
              onClick={handleSaveToLibrary}
            >
              <Icon as={BookmarkIcon} {...MENU_ITEM_ICON_PROPS} />
              <MenuItemBody hint={t('graphPreview.saveToLibraryHint')} label={t('graphPreview.saveToLibrary')} />
            </Menu.Item>
            <Menu.Item
              {...MENU_ITEM_LAYOUT}
              _disabled={DISABLED_PROPS}
              disabled={!isTemplatesLoaded}
              value="fork-into-project"
              onClick={handleForkIntoProject}
            >
              <Icon as={GitForkIcon} {...MENU_ITEM_ICON_PROPS} />
              <MenuItemBody hint={t('graphPreview.forkIntoProjectHint')} label={t('graphPreview.forkIntoProject')} />
            </Menu.Item>
            <Menu.Item {...MENU_ITEM_LAYOUT} value="download-json" onClick={handleDownloadJson}>
              <Icon as={DownloadIcon} {...MENU_ITEM_ICON_PROPS} />
              <MenuItemBody hint={t('graphPreview.downloadJsonHint')} label={t('graphPreview.downloadJson')} />
            </Menu.Item>
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
