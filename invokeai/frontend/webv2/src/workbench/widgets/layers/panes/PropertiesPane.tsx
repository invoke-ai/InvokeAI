import { Box, Flex, Stack } from '@chakra-ui/react';
import { Scrollable } from '@platform/ui/Scrollable';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { useCanvasActiveTool, useCanvasOperation } from '@workbench/widgets/canvas/engineStoreHooks';
import { PropertyGroup } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import {
  OPERATION_PRESENTATION_ADAPTERS,
  TOOL_PRESENTATION_ADAPTERS,
} from '@workbench/widgets/canvas/tool-presentation/toolAdapters';
import { useLayoutEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { GroupSelectedNotice } from './GroupSelectedNotice';
import { LayerSection } from './LayerSection';
import { createOperationSection } from './OperationSection';
import { PropertiesSection } from './PropertiesSection';

const OPERATION_SECTIONS = {
  filter: createOperationSection(OPERATION_PRESENTATION_ADAPTERS.filter),
  'select-object': createOperationSection(OPERATION_PRESENTATION_ADAPTERS['select-object']),
} as const;

/** Show active operation before tool forms; use existing engine stores/transactions without mirrored state. */
export const PropertiesPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();

  if (!engine) {
    return (
      <Flex align="center" color="fg.muted" fontSize="md" h="full" justify="center" p="4">
        {t('widgets.properties.noCanvas')}
      </Flex>
    );
  }

  return (
    <Scrollable h="full">
      <ConnectedProperties engine={engine} />
    </Scrollable>
  );
};

const ConnectedProperties = ({ engine }: { engine: CanvasEngineHandle }) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const operation = useCanvasOperation(engine);
  const running = operation.status === 'active' ? OPERATION_PRESENTATION_ADAPTERS[operation.identity.kind] : null;
  const RunningSection = operation.status === 'active' ? OPERATION_SECTIONS[operation.identity.kind] : null;
  const tool = TOOL_PRESENTATION_ADAPTERS[activeTool];
  const toolName = t(`widgets.canvas.tools.${tool.id}`);
  const Preview = tool.preview;
  const regionProps = { engine };
  const ToolActions = tool.actions;

  // `inert` will blur a focused tool control once an operation starts; hand focus to the operation's Cancel first. The
  // Operation section comes first, so its Cancel is the first match; focusing it scrolls it into view.
  const root = useRef<HTMLDivElement>(null);
  const toolSection = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (running && toolSection.current?.contains(document.activeElement)) {
      root.current?.querySelector<HTMLElement>('[data-pane-action="cancel"]')?.focus();
    }
  }, [running]);

  return (
    <Stack ref={root} gap="0">
      {RunningSection ? <RunningSection engine={engine} /> : null}
      {tool.groups.length > 0 || Preview || ToolActions ? (
        <PropertiesSection
          ref={toolSection}
          disabled={running !== null}
          subtitle={toolName}
          title={t('widgets.properties.sections.tool')}
        >
          {tool.paintsLeaf && !running ? <GroupSelectedNotice /> : null}
          {Preview ? <Preview engine={engine} /> : null}
          {tool.groups.map((group) => (
            // Keyed by GROUP id, not tool id: tools sharing a group keep its
            // DOM (and collapse state) alive across the tool switch.
            <PropertyGroup key={group.id} collapsible={group.collapsible} id={group.id} label={t(group.labelKey)}>
              <group.body {...regionProps} />
            </PropertyGroup>
          ))}
          {ToolActions ? (
            <Box pt="1">
              <ToolActions engine={engine} />
            </Box>
          ) : null}
        </PropertiesSection>
      ) : null}
      <LayerSection disabled={running !== null} />
    </Stack>
  );
};
