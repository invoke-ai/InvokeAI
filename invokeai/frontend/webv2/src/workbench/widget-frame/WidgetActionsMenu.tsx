import type { GraphContract } from '@workbench/graphContracts';
import type { Project } from '@workbench/projectContracts';
import type {
  GraphBearingSurfaceContract,
  WidgetInstanceRuntimeMeta,
  WidgetHeaderMenu,
  WidgetManifest,
  WidgetRuntimeApi,
  WorkbenchRegion,
} from '@workbench/widgetContracts';

import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { IconButton } from '@platform/ui';
import { createGraphBearingSurface } from '@workbench/graphSurfaces';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { shallowEqual, useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { GitBranchIcon, MoreHorizontalIcon, TargetIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The widget frame's shared header actions menu. It hosts the universal
 * graph-bearing actions (`Set Source`, `View Graph`) and any extra entries the
 * widget's manifest contributes via `headerMenu` — one menu per widget, so
 * widgets extend the frame instead of stacking their own menus and toolbars.
 */

const GraphPreviewDialog = lazy(() =>
  import('@features/workflow/preview').then((module) => ({ default: module.GraphPreviewDialog }))
);

const getPreviewGraph = async (
  project: Project,
  surface: GraphBearingSurfaceContract
): Promise<GraphContract | null> => {
  // The project graph compiles fresh for preview so `View Graph` always shows
  // what Invoke would run right now; widget graphs keep their last compile.
  if (surface.sourceId === 'workflow') {
    const [{ compileProjectGraph }, { getInvocationTemplatesSnapshot }] = await Promise.all([
      import('@features/workflow/graph'),
      import('@features/workflow/react'),
    ]);
    const templatesSnapshot = getInvocationTemplatesSnapshot();

    if (templatesSnapshot.status !== 'loaded') {
      return null;
    }

    try {
      return compileProjectGraph(project.projectGraph, templatesSnapshot.templates);
    } catch {
      return null;
    }
  }

  return project.widgetGraphs[surface.widgetId] ?? null;
};

const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const DISABLED_PROPS = { opacity: 0.4 };

const GraphSurfaceMenuItems = ({
  surface,
  onPreview,
}: {
  surface: GraphBearingSurfaceContract;
  onPreview: () => void;
}) => {
  const { t } = useTranslation();
  const activeSourceId = useActiveProjectSelector((project) => project.invocation.sourceId);
  const { generation } = useWorkbenchCommands();
  const isActiveSource = activeSourceId === surface.sourceId;
  const handleSetSource = useCallback(() => generation.setSource(surface.sourceId), [generation, surface.sourceId]);

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
        {t('common.graph')}
      </Menu.ItemGroupLabel>
      <Menu.Item
        value="set-source"
        disabled={isActiveSource || !surface.canSetSource}
        _disabled={DISABLED_PROPS}
        onClick={handleSetSource}
      >
        <Icon as={TargetIcon} boxSize="3.5" />
        <Menu.ItemText>{t('widgets.graph.setSource')}</Menu.ItemText>
        {isActiveSource ? (
          <Text color="fg.subtle" fontSize="2xs" ms="auto">
            {t('common.active')}
          </Text>
        ) : null}
      </Menu.Item>
      <Menu.Item value="view-graph" disabled={!surface.canPreviewGraph} onClick={onPreview}>
        <Icon as={GitBranchIcon} boxSize="3.5" />
        <Menu.ItemText>{t('widgets.graph.viewGraph')}</Menu.ItemText>
      </Menu.Item>
    </Menu.ItemGroup>
  );
};

export const WidgetActionsMenu = ({
  HeaderMenu,
  instance,
  manifest,
  region,
  runtime,
}: {
  HeaderMenu?: WidgetHeaderMenu;
  instance: WidgetInstanceRuntimeMeta;
  manifest: WidgetManifest;
  region: WorkbenchRegion;
  runtime: WidgetRuntimeApi;
}) => {
  const { t } = useTranslation();
  const activeProject = useActiveProjectSelector(
    (project) => ({ projectGraph: project.projectGraph, widgetGraphs: project.widgetGraphs }),
    shallowEqual
  ) as Project;
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [previewGraph, setPreviewGraph] = useState<GraphContract | null>(null);
  const label = resolveWidgetLabel(manifest, t);
  const surface = useMemo(
    () =>
      manifest.graphBearing?.surfaces.includes(region) ? createGraphBearingSurface(manifest, region, label) : null,
    [label, manifest, region]
  );
  const surfaceSourceId = surface?.sourceId;
  const handlePreview = useCallback(async () => {
    flushWorkbenchDrafts();
    if (surface) {
      setPreviewGraph(await getPreviewGraph(activeProject, surface));
    }
    setIsPreviewOpen(true);
  }, [activeProject, surface]);
  const positionHints =
    isPreviewOpen && surfaceSourceId === 'workflow'
      ? Object.fromEntries(activeProject.projectGraph.nodes.map((node) => [node.id, node.position]))
      : undefined;

  if (!surface && !HeaderMenu) {
    return null;
  }

  // The project graph mirrors the editable document, so the preview can reuse
  // the editor's node positions instead of auto-layouting.
  return (
    <>
      <Menu.Root positioning={MENU_POSITIONING}>
        <Menu.Trigger asChild>
          <IconButton aria-label={t('widgets.actionsLabel', { label })} color="fg.muted" size="2xs" variant="ghost">
            <MoreHorizontalIcon />
          </IconButton>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <Menu.Content minW="13rem">
              {surface ? <GraphSurfaceMenuItems surface={surface} onPreview={handlePreview} /> : null}
              {surface && HeaderMenu ? <Menu.Separator borderColor="border.subtle" /> : null}
              {HeaderMenu ? (
                <HeaderMenu instance={instance} manifest={manifest} region={region} runtime={runtime} />
              ) : null}
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      {surface && isPreviewOpen ? (
        <Suspense fallback={null}>
          <GraphPreviewDialog
            graph={previewGraph}
            graphId={surface.graphId}
            isOpen={isPreviewOpen}
            positionHints={positionHints}
            sourceId={surface.sourceId}
            title={surface.label}
            onOpenChange={setIsPreviewOpen}
          />
        </Suspense>
      ) : null}
    </>
  );
};
