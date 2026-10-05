import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  GraphPreviewNotice,
  GraphPreviewProvenance,
  GraphPreviewSourceState,
  GraphPreviewSummaryRow,
} from '@features/workflow/contracts';
import type { InvocationTemplatesSnapshot } from '@features/workflow/react';
import type { Project } from '@workbench/projectContracts';
import type { GraphBearingSurfaceContract } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';

import { compileGeneratePreviewGraph, getGenerateNodeProvenance } from '@features/generation/preview';
import { compileProjectGraph } from '@features/workflow/graph';
import { ForLoopGraphValidationError } from '@features/workflow/utility';
import { getDestinationLabel } from '@workbench/invocation';
import { getActiveProjectGraph } from '@workbench/projectWorkflows';
import { getProjectWidgetValues } from '@workbench/widgetState';

/**
 * Translate project/surface to preview data without effects; callers load models/templates before per-edit
 * compilation.
 */
export interface GraphPreviewSourceDeps {
  models: readonly ModelConfig[] | undefined;
  project: Project;
  surface: GraphBearingSurfaceContract;
  t: TFunction;
  templates: InvocationTemplatesSnapshot;
}

const buildGenerateSummaryRows = (settings: GenerateWidgetValues, t: TFunction): GraphPreviewSummaryRow[] => {
  const activeLoras = settings.loras.filter((lora) => lora.isEnabled).length;

  return [
    { id: 'model', label: t('graphPreview.model'), value: settings.model.name },
    { id: 'size', label: t('graphPreview.size'), value: `${settings.width} × ${settings.height}` },
    { id: 'steps', label: t('graphPreview.steps'), value: String(settings.steps) },
    { id: 'cfgScale', label: t('graphPreview.cfgScale'), value: String(settings.cfgScale) },
    { id: 'scheduler', label: t('graphPreview.scheduler'), value: settings.scheduler },
    ...(activeLoras > 0 ? [{ id: 'loras', label: t('graphPreview.loras'), value: String(activeLoras) }] : []),
    {
      id: 'seed',
      label: t('graphPreview.seed'),
      value: settings.seedMode === 'random' ? t('graphPreview.seedRandomValue') : String(settings.seed),
    },
  ];
};

const EMPTY_SOURCE_BASE: Pick<GraphPreviewSourceState, 'invalidReasons' | 'notices' | 'summaryRows'> = {
  invalidReasons: [],
  notices: [],
  summaryRows: [],
};

/** `destinationLabel` is filled in once by `buildGraphPreviewSource` for every source, below. */
type GraphPreviewSourceWithoutDestination = Omit<GraphPreviewSourceState, 'destinationLabel'>;

const buildWorkflowSource = (
  project: Project,
  templates: InvocationTemplatesSnapshot
): GraphPreviewSourceWithoutDestination => {
  if (templates.status !== 'loaded') {
    return { ...EMPTY_SOURCE_BASE, graph: null, isLive: true };
  }

  const document = getActiveProjectGraph(project);
  const positionHints = Object.fromEntries(document.nodes.map((node) => [node.id, node.position]));

  try {
    const graph = compileProjectGraph(document, templates.templates);

    return { ...EMPTY_SOURCE_BASE, graph, isLive: true, positionHints };
  } catch (error) {
    return {
      ...EMPTY_SOURCE_BASE,
      graph: null,
      invalidReasons: [
        error instanceof ForLoopGraphValidationError
          ? error.reason
          : error instanceof Error
            ? error.message
            : String(error),
      ],
      isLive: true,
      positionHints,
    };
  }
};

const buildGenerateSource = (
  project: Project,
  models: readonly ModelConfig[] | undefined,
  t: TFunction
): GraphPreviewSourceWithoutDestination => {
  let result: ReturnType<typeof compileGeneratePreviewGraph>;

  // Return compile failures as preview reasons instead of throwing through widget chrome.
  try {
    result = compileGeneratePreviewGraph({
      destination: project.invocation.destination,
      models: models ?? [],
      storedValues: getProjectWidgetValues(project, 'generate'),
      useCpuNoise: project.settings.useCpuNoise,
    });
  } catch (error) {
    return {
      ...EMPTY_SOURCE_BASE,
      graph: null,
      invalidReasons: [error instanceof Error ? error.message : String(error)],
      isLive: true,
    };
  }

  if (result.status === 'invalid') {
    return { ...EMPTY_SOURCE_BASE, graph: null, invalidReasons: result.reasons, isLive: true };
  }

  const { settings } = result;
  const isSeedRandomized = settings.seedMode === 'random';
  const notices: GraphPreviewNotice[] = isSeedRandomized
    ? [{ id: 'seed-random', message: t('graphPreview.seedRandomized'), nodeId: 'seed' }]
    : [];
  const resolvedInputOverrides = isSeedRandomized ? { seed: { value: t('graphPreview.seedRegenerated') } } : undefined;
  const getProvenance = (nodeId: string, fieldName: string): GraphPreviewProvenance | null => {
    const entry = getGenerateNodeProvenance(nodeId, fieldName);

    if (!entry) {
      return null;
    }

    if (entry.settingKey === 'seed' && isSeedRandomized) {
      return { label: t('graphPreview.provenance.seedRandom') };
    }

    return { label: t(entry.labelKey) };
  };

  return {
    getProvenance,
    graph: result.graph,
    invalidReasons: [],
    isLive: true,
    notices,
    resolvedInputOverrides,
    summaryRows: buildGenerateSummaryRows(settings, t),
  };
};

const buildWidgetGraphSource = (
  project: Project,
  surface: GraphBearingSurfaceContract
): GraphPreviewSourceWithoutDestination => ({
  ...EMPTY_SOURCE_BASE,
  graph: project.widgetGraphs[surface.widgetId] ?? null,
  isLive: false,
});

export const buildGraphPreviewSource = ({
  models,
  project,
  surface,
  t,
  templates,
}: GraphPreviewSourceDeps): GraphPreviewSourceState => {
  const destinationLabel = getDestinationLabel(project.invocation.destination);

  switch (surface.sourceId) {
    case 'workflow':
      return { ...buildWorkflowSource(project, templates), destinationLabel };
    case 'generate':
      return { ...buildGenerateSource(project, models, t), destinationLabel };
    case 'upscale':
    case 'video':
    case 'canvas':
      return { ...buildWidgetGraphSource(project, surface), destinationLabel };
  }
};
