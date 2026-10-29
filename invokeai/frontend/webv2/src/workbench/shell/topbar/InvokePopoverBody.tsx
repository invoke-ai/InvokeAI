import type { InvocationSourceId } from '@workbench/invocationContracts';

import { HStack, Icon, Kbd, Popover, SegmentGroup, Separator, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { WidgetIcon } from '@workbench/iconResolver';
import { ROUTE_LABEL_KEYS } from '@workbench/invocation';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ArrowRightIcon, CircleAlertIcon, WifiOffIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationIssue, InvocationState } from './useInvocationState';

import { useInvokeIssueActions } from './invokeIssueActions';
import { TopbarShortcutKeys } from './TopbarShortcutKeys';

/** The Invoke card: what runs, where it goes, and each issue in the way with its fixes. Loads on first hover or focus. */
export const InvokePopoverBody = ({
  frontShortcut,
  onJump,
  shortcutParts,
  state,
}: {
  frontShortcut: string | null;
  /** An action that leads elsewhere closes the list, leaving focus where the action put it. */
  onJump: () => void;
  shortcutParts: string[] | null;
  state: InvocationState;
}) => {
  const { t } = useTranslation();
  const { batchCount, invocation, isPreparing, isValid, issues, promptExpansion, workflowBatchSize } = state;
  const source = t(ROUTE_LABEL_KEYS[invocation.sourceId]);
  const destination = t(ROUTE_LABEL_KEYS[invocation.destination]);
  const promptCount = promptExpansion.count;
  const runs = t('topbar.invoke.summary.runs', { count: batchCount });
  const summary =
    invocation.sourceId === 'generate' || invocation.sourceId === 'upscale' || invocation.sourceId === 'video'
      ? promptExpansion.isLoading
        ? t('topbar.invoke.expandingPrompts')
        : t('topbar.invoke.summary.prompts', {
            generations: t('topbar.invoke.summary.generations', { count: promptCount * batchCount }),
            iterations: t('topbar.invoke.summary.iterations', { count: batchCount }),
            prompts: t('topbar.invoke.summary.promptCount', { count: promptCount }),
          })
      : workflowBatchSize === undefined
        ? t('topbar.invoke.summary.runsOnly', {
            generations: t('topbar.invoke.summary.generations', { count: batchCount }),
            runs,
            source,
          })
        : workflowBatchSize === null
          ? t('topbar.invoke.summary.batchPending', { runs, source })
          : t('topbar.invoke.summary.batch', {
              generations: t('topbar.invoke.summary.generations', { count: batchCount * workflowBatchSize }),
              items: t('topbar.invoke.summary.batchItems', { count: workflowBatchSize }),
              runs,
              source,
            });

  return (
    <Stack gap="2" maxW="22rem" minW="17rem">
      <HStack justify="space-between">
        <Popover.Title fontSize="sm" fontWeight="700">
          {isPreparing
            ? t('topbar.invoke.preparing')
            : isValid
              ? t('topbar.invoke.addToQueue')
              : t('topbar.invoke.unableToQueue')}
        </Popover.Title>
        {shortcutParts ? (
          <Kbd variant="subtle">
            <TopbarShortcutKeys parts={shortcutParts} />
          </Kbd>
        ) : null}
      </HStack>
      <InvokeRoute destination={destination} state={state} />
      <Text color="fg.muted" fontSize="xs">
        {summary}
      </Text>
      <Separator borderColor="border.subtle" />
      {isPreparing ? (
        <Text color="fg.muted" fontSize="sm">
          {t('topbar.invoke.preparing')}
        </Text>
      ) : issues.length > 0 ? (
        <InvokeIssueList issues={issues} onJump={onJump} />
      ) : (
        <Text color="fg.muted" fontSize="sm">
          {t('topbar.invoke.addingImagesTo', { destination })}
        </Text>
      )}
      <Text color="fg.subtle" fontSize="xs">
        {frontShortcut === null
          ? t('topbar.invoke.invokeFrontHintNoShortcut')
          : t('topbar.invoke.invokeFrontHint', { shortcut: frontShortcut })}
      </Text>
    </Stack>
  );
};

/**
 * Names what Invoke runs and where results go, and switches the source in place: among the sources this layout holds,
 * plus the current one when it is missing, so the choice is never between things the user cannot see.
 */
const InvokeRoute = ({ destination, state }: { destination: string; state: InvocationState }) => {
  const { t } = useTranslation();
  const { generation } = useWorkbenchCommands();
  const { invocation, placedTypeIds, sources } = state;
  const options = sources.filter(
    (candidate) => placedTypeIds.has(candidate.typeId) || candidate.sourceId === invocation.sourceId
  );
  const handleChange = useCallback(
    ({ value }: { value: string | null }) => {
      if (value) {
        generation.setSource(value as InvocationSourceId);
      }
    },
    [generation]
  );

  return (
    <Stack gap="1">
      <Text color="fg.subtle" fontSize="xs" fontWeight="700" textTransform="uppercase">
        {t('topbar.invoke.runs')}
      </Text>
      <HStack gap="2" wrap="wrap">
        {options.length > 1 ? (
          <SegmentGroup.Root
            aria-label={t('topbar.routing.source')}
            size="xs"
            value={invocation.sourceId}
            onValueChange={handleChange}
          >
            <SegmentGroup.Indicator />
            {options.map((option) => (
              <SegmentGroup.Item key={option.sourceId} value={option.sourceId}>
                <SegmentGroup.ItemHiddenInput />
                <SegmentGroup.ItemText alignItems="center" display="flex" fontSize="xs" gap="1">
                  <WidgetIcon boxSize="3" icon={getWidgetById(option.typeId)?.manifest.icon} />
                  {t(ROUTE_LABEL_KEYS[option.sourceId])}
                </SegmentGroup.ItemText>
              </SegmentGroup.Item>
            ))}
          </SegmentGroup.Root>
        ) : (
          <Text fontSize="sm" fontWeight="600">
            {t(ROUTE_LABEL_KEYS[invocation.sourceId])}
          </Text>
        )}
        <HStack color="fg.muted" gap="1">
          <Icon aria-hidden as={ArrowRightIcon} boxSize="3" />
          <Text fontSize="sm">{t('topbar.invoke.resultsTo', { destination })}</Text>
        </HStack>
      </HStack>
    </Stack>
  );
};

const InvokeIssueList = ({ issues, onJump }: { issues: readonly InvocationIssue[]; onJump: () => void }) => {
  const getActions = useInvokeIssueActions();

  return (
    <Stack as="ul" gap="2" listStyleType="none" m="0" p="0">
      {issues.map((issue, index) => {
        const actions = getActions(issue);

        return (
          // Two issues can read the same (two nodes missing the same input), so the position keeps keys apart.
          // oxlint-disable-next-line react/no-array-index-key -- the list is rebuilt whole on every change.
          <Stack key={`${index}:${issue.message}`} as="li" gap="1">
            <HStack align="start" gap="1.5">
              <Icon
                aria-hidden
                as={issue.target.kind === 'connection' ? WifiOffIcon : CircleAlertIcon}
                boxSize="3"
                color={issue.target.kind === 'connection' ? 'fg.muted' : 'fg.error'}
                flexShrink={0}
                mt="0.5"
              />
              <Text color="fg" fontSize="sm" lineHeight="1.4" textWrap="pretty">
                {issue.message}
              </Text>
            </HStack>
            {actions.length > 0 ? (
              <HStack gap="1" ps="4.5" wrap="wrap">
                {actions.map((action) => (
                  <Button
                    key={action.id}
                    size="xs"
                    data-issue-action=""
                    variant="outline"
                    // oxlint-disable-next-line react-perf/jsx-no-new-function-as-prop -- a few buttons per shown issue.
                    onClick={() => {
                      action.run();
                      if (!action.stays) {
                        onJump();
                      }
                    }}
                  >
                    {action.label}
                  </Button>
                ))}
              </HStack>
            ) : null}
          </Stack>
        );
      })}
    </Stack>
  );
};
