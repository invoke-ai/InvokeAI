import type {
  CanvasToolOptionsEngine,
  OperationPropertyForm,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
/* oxlint-disable react-perf/jsx-no-jsx-as-prop */

import { Box, Flex, Icon, Spinner, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { PropertyGroup } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { InfoIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { PropertiesSection } from './PropertiesSection';

const OPERATION_TITLE_KEYS = {
  filter: 'widgets.layers.rasterFilter.title',
  'select-object': 'widgets.layers.selectObject.title',
} as const;

/**
 * The running operation's section: its groups and actions, the busy phase beside the title, and any error last, where it
 * can grow without moving a control. One component per form, so each calls one status hook for its lifetime.
 */
export const createOperationSection = (form: OperationPropertyForm) => {
  const useStatus = form.useStatus;
  const Actions = form.actions;
  const OperationSection = ({ engine }: { engine: CanvasToolOptionsEngine }) => {
    const { t } = useTranslation();
    const status = useStatus(engine);
    const error = status?.error ?? null;
    const detail = error?.detail?.trim();
    const busyText = error ? null : (status?.busyText ?? null);
    return (
      <PropertiesSection
        subtitle={t(OPERATION_TITLE_KEYS[form.kind])}
        subtitleDetail={status?.source}
        title={t('widgets.properties.sections.operation')}
        trailing={
          // Mounted for as long as the operation runs, so the phase it reports is announced.
          <Text aria-live="polite" color="fg.muted" flex="1 1 0" minW="0" role="status" textAlign="end" truncate>
            {busyText ? (
              // The tooltip shows the whole phase where a narrow pane truncates it.
              <Tooltip content={busyText}>
                <span>
                  {/* Inline in the text line, so the header keeps its height and baseline while busy. */}
                  <Spinner me="1.5" size="inherit" verticalAlign="-0.125em" />
                  {busyText}
                </span>
              </Tooltip>
            ) : null}
          </Text>
        }
      >
        {form.groups.map((group) => (
          <PropertyGroup key={group.id} collapsible={group.collapsible} id={group.id} label={t(group.labelKey)}>
            <group.body engine={engine} />
          </PropertyGroup>
        ))}
        <Box pt="1">
          <Actions engine={engine} />
        </Box>
        {error ? (
          <Flex align="flex-start" color="fg.error" gap="1">
            <Text aria-live="assertive" flex="1" minW="0" py="1" role="alert">
              {error.text}
            </Text>
            {detail && detail !== error.text ? (
              <IconButton
                aria-label={t('widgets.layers.selectObject.technicalDetails')}
                color="fg.muted"
                flexShrink={0}
                size="xs"
                tooltip={detail}
                variant="ghost"
              >
                <Icon as={InfoIcon} boxSize="3.5" />
              </IconButton>
            ) : null}
          </Flex>
        ) : null}
      </PropertiesSection>
    );
  };
  return OperationSection;
};
