import { Flex, HStack, Icon, Splitter } from '@chakra-ui/react';
import { ensureInvocationTemplatesLoaded } from '@features/workflow/react';
import { useWorkflowHostCommands, useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, Scrollable, Tabs } from '@platform/ui';
import { EyeIcon, PencilIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { FormBuilderTab } from './FormBuilderTab';
import { LinearFormView } from './LinearFormView';
import { NodeInspector } from './NodeInspector';
import { WorkflowDetailsTab } from './WorkflowDetailsTab';
import { WorkflowJsonTab } from './WorkflowJsonTab';

/**
 * The workflow widget's side panel — the Linear UI. View mode runs the
 * project graph through its form; Edit mode opens the legacy-style tab set
 * (Form builder / Details / JSON) with the node inspector pinned below.
 * Panel mode, tab, and inspector height live in the widget's own state.
 */

type PanelMode = 'view' | 'edit';
type EditTab = 'form' | 'details' | 'json';
type PanelModeItem = {
  labelKey: string;
  icon: typeof EyeIcon;
  mode: PanelMode;
};

/** Inspector share of the splitter, as a percentage of the panel height. */
const DEFAULT_INSPECTOR_SIZE_PCT = 35;
const SPLITTER_PANELS = [
  { id: 'content', minSize: 25 },
  { id: 'inspector', minSize: 12 },
];
const PANEL_MODES: PanelModeItem[] = [
  { labelKey: 'common.view', icon: EyeIcon, mode: 'view' },
  { labelKey: 'common.edit', icon: PencilIcon, mode: 'edit' },
];

export interface WorkflowPanelState {
  editTab: EditTab;
  inspectorSizePct: number;
  mode: PanelMode;
}

const getPanelMode = (values: Record<string, unknown>): PanelMode => (values.panelMode === 'edit' ? 'edit' : 'view');

const getEditTab = (values: Record<string, unknown>): EditTab =>
  values.editTab === 'details' || values.editTab === 'json' ? values.editTab : 'form';

const getInspectorSizePct = (values: Record<string, unknown>): number =>
  typeof values.inspectorSizePct === 'number' && Number.isFinite(values.inspectorSizePct)
    ? Math.min(75, Math.max(12, values.inspectorSizePct))
    : DEFAULT_INSPECTOR_SIZE_PCT;

export const getWorkflowPanelState = (values: Record<string, unknown>): WorkflowPanelState => ({
  editTab: getEditTab(values),
  inspectorSizePct: getInspectorSizePct(values),
  mode: getPanelMode(values),
});

export const areWorkflowPanelStatesEqual = (left: WorkflowPanelState, right: WorkflowPanelState): boolean =>
  left.mode === right.mode && left.editTab === right.editTab && left.inspectorSizePct === right.inspectorSizePct;

export const PanelModeToggle = ({
  mode: currentMode,
  onChange,
}: {
  mode: PanelMode;
  onChange: (mode: PanelMode) => void;
}) => {
  const { t } = useTranslation();

  return (
    <HStack
      aria-label="Workflow panel mode"
      borderColor="border"
      borderRadius="md"
      borderWidth="1px"
      gap="0"
      overflow="hidden"
      role="group"
    >
      {PANEL_MODES.map(({ labelKey, icon, mode: itemMode }) => (
        <PanelModeButton
          key={itemMode}
          currentMode={currentMode}
          icon={icon}
          label={t(labelKey)}
          mode={itemMode}
          onChange={onChange}
        />
      ))}
    </HStack>
  );
};

const PanelModeButton = ({
  currentMode,
  icon,
  label,
  mode,
  onChange,
}: {
  currentMode: PanelMode;
  icon: typeof EyeIcon;
  label: string;
  mode: PanelMode;
  onChange: (mode: PanelMode) => void;
}) => {
  const onClick = useCallback(() => onChange(mode), [mode, onChange]);

  return (
    <Button
      aria-pressed={mode === currentMode}
      borderRadius="0"
      colorPalette={mode === currentMode ? 'accent' : 'bg'}
      size="xs"
      variant={mode === currentMode ? 'solid' : 'ghost'}
      onClick={onClick}
    >
      <Icon as={icon} boxSize="3" />
      {label}
    </Button>
  );
};

export const WorkflowLinearPanel = () => {
  const { t } = useTranslation();
  const { editTab, inspectorSizePct, mode } = useWorkflowProjectSelector((project) =>
    getWorkflowPanelState(project.workflowValues)
  );
  const { widgets } = useWorkflowHostCommands();

  useMountEffect(() => {
    ensureInvocationTemplatesLoaded();
  });

  const patchValues = useCallback(
    (values: Record<string, unknown>) => widgets.patchValues('workflow', values),
    [widgets]
  );
  const onPanelModeChange = useCallback((panelMode: PanelMode) => patchValues({ panelMode }), [patchValues]);
  const onEditTabChange = useCallback(
    (event: { value: string }) => patchValues({ editTab: event.value }),
    [patchValues]
  );

  return (
    <Flex direction="column" flex="1" h="full" minH="0">
      <HStack flexShrink={0} justify="space-between" px="2" h={10} borderBottomWidth={1}>
        <PanelModeToggle mode={mode} onChange={onPanelModeChange} />
        {mode === 'edit' ? (
          <Tabs.Root size="sm" value={editTab} variant="outline" mb="-1" onValueChange={onEditTabChange}>
            <Tabs.List>
              <Tabs.Trigger value="form" fontSize="2xs">
                {t('widgets.workflow.form')}
              </Tabs.Trigger>
              <Tabs.Trigger value="details" fontSize="2xs">
                {t('widgets.workflow.details')}
              </Tabs.Trigger>
              <Tabs.Trigger value="json" fontSize="2xs">
                {t('common.json')}
              </Tabs.Trigger>
            </Tabs.List>
          </Tabs.Root>
        ) : null}
      </HStack>
      {mode === 'view' ? (
        <WorkflowLinearViewContent />
      ) : (
        <WorkflowLinearEditContent editTab={editTab} inspectorSizePct={inspectorSizePct} patchValues={patchValues} />
      )}
    </Flex>
  );
};

const WorkflowLinearViewContent = () => {
  const { t } = useTranslation();
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);

  return (
    <Scrollable flex="1" label={t('widgets.workflow.panelContent')} minH="0">
      <LinearFormView projectGraph={projectGraph} />
    </Scrollable>
  );
};

const WorkflowLinearEditContent = ({
  editTab,
  inspectorSizePct,
  patchValues,
}: {
  editTab: EditTab;
  inspectorSizePct: number;
  patchValues: (values: Record<string, unknown>) => void;
}) => {
  const { t } = useTranslation();
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);
  const defaultSize = useMemo(() => [100 - inspectorSizePct, inspectorSizePct], [inspectorSizePct]);
  const onResizeEnd = useCallback(
    (details: { size: number[] }) => {
      const inspectorSizePct = details.size[1];

      if (typeof inspectorSizePct === 'number') {
        patchValues({ inspectorSizePct });
      }
    },
    [patchValues]
  );

  return (
    <Splitter.Root
      defaultSize={defaultSize}
      flex="1"
      gap="0"
      minH="0"
      orientation="vertical"
      panels={SPLITTER_PANELS}
      onResizeEnd={onResizeEnd}
    >
      <Splitter.Panel id="content" minH="0" minW="0" overflow="hidden">
        {editTab === 'json' ? (
          <WorkflowJsonTab projectGraph={projectGraph} />
        ) : (
          <Scrollable flex="1" h="full" label={t('widgets.workflow.panelContent')} minH="0" minW="0" w="full">
            {editTab === 'form' ? (
              <FormBuilderTab projectGraph={projectGraph} />
            ) : (
              <WorkflowDetailsTab metadata={projectGraph} />
            )}
          </Scrollable>
        )}
      </Splitter.Panel>
      <Splitter.ResizeTrigger aria-label={t('widgets.workflow.resizeNodeInspector')} id="content:inspector" />
      <Splitter.Panel id="inspector" minH="0" overflow="hidden">
        <NodeInspector projectGraph={projectGraph} />
      </Splitter.Panel>
    </Splitter.Root>
  );
};
