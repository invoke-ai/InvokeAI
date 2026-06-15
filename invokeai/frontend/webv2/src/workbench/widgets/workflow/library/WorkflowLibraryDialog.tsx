import type { ProjectGraphState } from '@workbench/workflows/types';

import { Box, Dialog, HStack, Input, Portal, SegmentGroup, Stack, Tabs, Text } from '@chakra-ui/react';
import { getApiErrorMessage } from '@workbench/backend/http';
import { documentToPreviewGraph, GraphPreviewFlow } from '@workbench/components/GraphPreviewFlow';
import { Button, CloseButton } from '@workbench/components/ui/Button';
import { ConfirmDialog } from '@workbench/components/ui/ConfirmDialog';
import { JsonPreview } from '@workbench/components/ui/JsonPreview';
import { Scrollable } from '@workbench/components/ui/Scrollable';
import { useNotify } from '@workbench/useNotify';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import {
  createLibraryWorkflow,
  deleteLibraryWorkflow,
  touchLibraryWorkflowOpenedAt,
  updateLibraryWorkflow,
  type WorkflowLibraryCategory,
  type WorkflowLibraryListItem,
} from '@workbench/workflows/api';
import {
  getCachedWorkflowPage,
  getLibraryWorkflowCached,
  invalidateWorkflowLibraryCache,
  listLibraryWorkflowsCached,
} from '@workbench/workflows/libraryCache';
import { parseWorkflowJson, serializeWorkflowJson } from '@workbench/workflows/workflowJson';
import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react';

/**
 * Backend workflow library browser: list/search user and default workflows,
 * preview one (read-only nodes / JSON) before loading it into the project
 * graph (with an automatic graph-history snapshot of the current graph), and
 * save the project graph back to the library. Lists serve from the session
 * cache instantly and revalidate in the background.
 */

interface WorkflowPreviewState {
  document: ProjectGraphState;
  item: WorkflowLibraryListItem;
  raw: Record<string, unknown>;
}

const WorkflowPreviewPane = ({
  isLoadingWorkflow,
  preview,
  onBack,
  onLoad,
}: {
  isLoadingWorkflow: boolean;
  preview: WorkflowPreviewState;
  onBack: () => void;
  onLoad: () => void;
}) => {
  const [mode, setMode] = useState<'nodes' | 'json'>('nodes');
  const { graph, positionHints } = documentToPreviewGraph(preview.document);

  return (
    <Stack gap="2">
      <HStack gap="2" justify="space-between">
        <HStack gap="2" minW="0">
          <Button size="2xs" variant="ghost" onClick={onBack}>
            ← Back
          </Button>
          <Text fontSize="xs" fontWeight="600" minW="0" truncate>
            {preview.item.name || 'Untitled Workflow'}
          </Text>
        </HStack>
        <HStack flexShrink={0} gap="2">
          <SegmentGroup.Root
            size="xs"
            value={mode}
            onValueChange={(event) => setMode(event.value === 'json' ? 'json' : 'nodes')}
          >
            <SegmentGroup.Indicator />
            <SegmentGroup.Items
              items={[
                { label: 'Nodes', value: 'nodes' },
                { label: 'JSON', value: 'json' },
              ]}
            />
          </SegmentGroup.Root>
          <Button loading={isLoadingWorkflow} size="2xs" onClick={onLoad}>
            Load
          </Button>
        </HStack>
      </HStack>
      {mode === 'nodes' ? (
        <Box h="24rem">
          <GraphPreviewFlow graph={graph} positionHints={positionHints} />
        </Box>
      ) : (
        <JsonPreview label={`${preview.item.name} JSON`} maxH="24rem" value={preview.raw} />
      )}
    </Stack>
  );
};

export const WorkflowLibraryDialog = ({
  isOpen,
  onOpenChange,
}: {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
}) => {
  const projectGraph = useActiveProjectSelector((project) => project.projectGraph);
  const dispatch = useWorkbenchDispatch();
  const notify = useNotify();
  const [category, setCategory] = useState<WorkflowLibraryCategory>('user');
  const [searchTerm, setSearchTerm] = useState('');
  const [items, setItems] = useState<WorkflowLibraryListItem[]>([]);
  const [pages, setPages] = useState(0);
  const [page, setPage] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingWorkflowId, setLoadingWorkflowId] = useState<string | null>(null);
  const [previewingWorkflowId, setPreviewingWorkflowId] = useState<string | null>(null);
  const [preview, setPreview] = useState<WorkflowPreviewState | null>(null);
  const [pendingDelete, setPendingDelete] = useState<WorkflowLibraryListItem | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const hasAutoSwitchedRef = useRef(false);
  const refreshTokenRef = useRef(0);

  const refresh = useCallback(
    (nextPage: number) => {
      const params = { category, page: nextPage, query: searchTerm };
      const cached = getCachedWorkflowPage(params);
      const token = (refreshTokenRef.current += 1);

      // A cached page renders instantly; the fetch below only revalidates it.
      if (cached) {
        setItems(cached.items);
        setPage(cached.page);
        setPages(cached.pages);
      } else {
        setIsLoading(true);
      }

      listLibraryWorkflowsCached(params)
        .then((result) => {
          if (token !== refreshTokenRef.current) {
            return;
          }

          setItems(result.items);
          setPage(result.page);
          setPages(result.pages);

          // A fresh install has no user workflows; land on the bundled
          // defaults instead of an empty tab. Once per dialog open.
          if (category === 'user' && result.total === 0 && !searchTerm && !hasAutoSwitchedRef.current) {
            hasAutoSwitchedRef.current = true;
            setCategory('default');
          }
        })
        .catch((error: unknown) => {
          if (cached) {
            return; // Stale-but-rendered beats an error toast on revalidation.
          }

          notify.error('Workflow library unavailable', getApiErrorMessage(error, 'Failed to list workflows.'));
        })
        .finally(() => {
          if (token === refreshTokenRef.current) {
            setIsLoading(false);
          }
        });
    },
    [category, notify, searchTerm]
  );

  useEffect(() => {
    if (isOpen) {
      refresh(0);
    } else {
      hasAutoSwitchedRef.current = false;
      setPreview(null);
    }
  }, [isOpen, refresh]);

  const fetchParsedWorkflow = async (item: WorkflowLibraryListItem) => {
    const raw = await getLibraryWorkflowCached(item.workflow_id);
    const { document, warnings } = parseWorkflowJson(raw);

    return { document, raw, warnings };
  };

  const applyLoadedWorkflow = (item: WorkflowLibraryListItem, document: ProjectGraphState, warnings: string[]) => {
    dispatch({ document, label: `Loaded "${item.name}" from library`, type: 'replaceProjectGraph' });

    for (const warning of warnings) {
      notify.info('Workflow load warning', warning);
    }

    void touchLibraryWorkflowOpenedAt(item.workflow_id).catch(() => {
      // Recency bookkeeping only; loading already succeeded.
    });
    onOpenChange(false);
  };

  const loadWorkflow = async (item: WorkflowLibraryListItem) => {
    setLoadingWorkflowId(item.workflow_id);

    try {
      const { document, warnings } = await fetchParsedWorkflow(item);

      applyLoadedWorkflow(item, document, warnings);
    } catch (error) {
      notify.error('Failed to load workflow', getApiErrorMessage(error, `Could not load "${item.name}".`));
    } finally {
      setLoadingWorkflowId(null);
    }
  };

  const previewWorkflow = async (item: WorkflowLibraryListItem) => {
    setPreviewingWorkflowId(item.workflow_id);

    try {
      const { document, raw } = await fetchParsedWorkflow(item);

      setPreview({ document, item, raw });
    } catch (error) {
      notify.error('Failed to preview workflow', getApiErrorMessage(error, `Could not preview "${item.name}".`));
    } finally {
      setPreviewingWorkflowId(null);
    }
  };

  const saveToLibrary = async (asNew: boolean) => {
    setIsSaving(true);

    try {
      const serialized = serializeWorkflowJson(projectGraph);

      if (!asNew && projectGraph.libraryWorkflowId) {
        await updateLibraryWorkflow(projectGraph.libraryWorkflowId, serialized);
        notify.success('Workflow saved', `Updated "${projectGraph.name || 'Untitled Workflow'}" in the library.`);
      } else {
        const workflowId = await createLibraryWorkflow(serialized);

        dispatch({ libraryWorkflowId: workflowId, type: 'setProjectGraphLibraryBinding' });
        notify.success('Workflow saved', `Saved "${projectGraph.name || 'Untitled Workflow'}" to the library.`);
      }

      invalidateWorkflowLibraryCache();
      refresh(page);
    } catch (error) {
      notify.error('Failed to save workflow', getApiErrorMessage(error, 'The workflow could not be saved.'));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <>
      <Dialog.Root open={isOpen} size="lg" onOpenChange={(event) => onOpenChange(event.open)}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" color="fg">
              <Dialog.Header>
                <Dialog.Title fontSize="sm" fontWeight="700">
                  Workflow Library
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                {preview ? (
                  <WorkflowPreviewPane
                    isLoadingWorkflow={loadingWorkflowId === preview.item.workflow_id}
                    preview={preview}
                    onBack={() => setPreview(null)}
                    onLoad={() => applyLoadedWorkflow(preview.item, preview.document, [])}
                  />
                ) : (
                  <Stack gap="3">
                    <HStack gap="2">
                      <Tabs.Root
                        size="sm"
                        value={category}
                        variant="enclosed"
                        onValueChange={(event) => setCategory(event.value as WorkflowLibraryCategory)}
                      >
                        <Tabs.List>
                          <Tabs.Trigger fontSize="xs" value="user">
                            Yours
                          </Tabs.Trigger>
                          <Tabs.Trigger fontSize="xs" value="default">
                            Defaults
                          </Tabs.Trigger>
                        </Tabs.List>
                      </Tabs.Root>
                      <Input
                        aria-label="Search workflows"
                        placeholder="Search workflows…"
                        size="xs"
                        value={searchTerm}
                        onChange={(event: ChangeEvent<HTMLInputElement>) => setSearchTerm(event.currentTarget.value)}
                      />
                    </HStack>
                    <Scrollable label="Workflow library results" maxH="20rem" minH="8rem">
                      <Stack gap="1" minW="0" w="full">
                        {items.map((item) => (
                          <Box
                            key={item.workflow_id}
                            _hover={{ bg: 'bg.emphasized' }}
                            alignItems="center"
                            display="grid"
                            gap="2"
                            gridTemplateColumns={
                              category === 'user' ? 'minmax(0, 1fr) auto auto auto' : 'minmax(0, 1fr) auto auto'
                            }
                            minW="0"
                            overflow="hidden"
                            px="1"
                            rounded="md"
                            transition="background 0.12s ease"
                            w="full"
                          >
                            <Box
                              as="button"
                              _focusVisible={{ outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' }}
                              aria-disabled={previewingWorkflowId !== null}
                              cursor="pointer"
                              flex="1"
                              minW="0"
                              px="1"
                              py="1.5"
                              rounded="md"
                              textAlign="start"
                              title={`Preview "${item.name || 'Untitled Workflow'}"`}
                              onClick={() => {
                                if (previewingWorkflowId === null) {
                                  void previewWorkflow(item);
                                }
                              }}
                            >
                              <Stack gap="0" minW="0">
                                <Text fontSize="xs" fontWeight="600" truncate>
                                  {item.name || 'Untitled Workflow'}
                                </Text>
                                {item.description ? (
                                  <Text color="fg.subtle" fontSize="2xs" truncate>
                                    {item.description}
                                  </Text>
                                ) : null}
                              </Stack>
                            </Box>
                            <Button
                              flexShrink={0}
                              loading={previewingWorkflowId === item.workflow_id}
                              size="2xs"
                              variant="outline"
                              onClick={() => void previewWorkflow(item)}
                            >
                              Preview
                            </Button>
                            <Button
                              flexShrink={0}
                              loading={loadingWorkflowId === item.workflow_id}
                              size="2xs"
                              variant="outline"
                              onClick={() => void loadWorkflow(item)}
                            >
                              Load
                            </Button>
                            {category === 'user' ? (
                              <Button
                                colorPalette="red"
                                flexShrink={0}
                                size="2xs"
                                variant="ghost"
                                onClick={() => setPendingDelete(item)}
                              >
                                Delete
                              </Button>
                            ) : null}
                          </Box>
                        ))}
                        {!isLoading && items.length === 0 ? (
                          <Text color="fg.subtle" fontSize="2xs" px="2" py="4" textAlign="center">
                            {category === 'user'
                              ? 'No saved workflows yet. Save the project graph below to start your library.'
                              : 'No default workflows are installed on this backend.'}
                          </Text>
                        ) : null}
                        {isLoading ? (
                          <Text color="fg.subtle" fontSize="2xs" px="2" py="1.5">
                            Loading workflows…
                          </Text>
                        ) : null}
                      </Stack>
                    </Scrollable>
                    {pages > 1 ? (
                      <HStack gap="2" justify="center">
                        <Button disabled={page === 0} size="2xs" variant="ghost" onClick={() => refresh(page - 1)}>
                          Previous
                        </Button>
                        <Text color="fg.subtle" fontSize="2xs">
                          Page {page + 1} of {pages}
                        </Text>
                        <Button
                          disabled={page >= pages - 1}
                          size="2xs"
                          variant="ghost"
                          onClick={() => refresh(page + 1)}
                        >
                          Next
                        </Button>
                      </HStack>
                    ) : null}
                    <Box borderColor="border.subtle" borderTopWidth="1px" pt="3">
                      <HStack gap="2" justify="space-between">
                        <Text color="fg.muted" fontSize="2xs" minW="0" truncate>
                          Project graph: {projectGraph.name || 'Untitled Workflow'}
                          {projectGraph.libraryWorkflowId ? ' (linked to library)' : ''}
                        </Text>
                        <HStack flexShrink={0} gap="2">
                          {projectGraph.libraryWorkflowId ? (
                            <Button
                              loading={isSaving}
                              size="2xs"
                              variant="outline"
                              onClick={() => void saveToLibrary(false)}
                            >
                              Save
                            </Button>
                          ) : null}
                          <Button
                            loading={isSaving}
                            size="2xs"
                            variant="outline"
                            onClick={() => void saveToLibrary(true)}
                          >
                            Save as new
                          </Button>
                        </HStack>
                      </HStack>
                    </Box>
                  </Stack>
                )}
              </Dialog.Body>
              <Dialog.CloseTrigger asChild>
                <CloseButton color="fg.muted" size="sm" />
              </Dialog.CloseTrigger>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      <ConfirmDialog
        body={`Delete "${pendingDelete?.name || 'Untitled Workflow'}" from the workflow library? This cannot be undone.`}
        confirmLabel="Delete"
        isOpen={pendingDelete !== null}
        title="Delete workflow"
        onClose={() => setPendingDelete(null)}
        onConfirm={async () => {
          if (!pendingDelete) {
            return;
          }

          try {
            await deleteLibraryWorkflow(pendingDelete.workflow_id);
            invalidateWorkflowLibraryCache();
            refresh(page);
          } catch (error) {
            notify.error('Failed to delete workflow', getApiErrorMessage(error, 'The workflow could not be deleted.'));
          }
        }}
      />
    </>
  );
};
