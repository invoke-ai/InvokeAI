import type { GenerateComponentValueKey } from '@features/generation/core/baseGenerationPolicies';
import type {
  GenerateSectionId,
  GenerationIssue,
  GenerationIssueField,
  GenerationIssueFix,
} from '@features/generation/core/generationIssues';
import type { ReactNode } from 'react';

import { HStack, Icon, Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { describeGenerationIssue, describeGenerationIssueFix } from '@features/generation/core/generationIssues';
import { Button, IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { CircleAlertIcon } from 'lucide-react';
import { createContext, use, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

interface GenerateIssuesValue {
  issues: readonly GenerationIssue[];
  applyFix: (fix: GenerationIssueFix) => void;
}

const GenerateIssuesContext = createContext<GenerateIssuesValue | null>(null);
/** How many Invoke-blocking issues each section holds, by section id, for the section titles. */
const SectionIssueCountsContext = createContext<ReadonlyMap<string, number> | null>(null);

const countIssuesBySection = (sections: readonly (string | undefined)[]): ReadonlyMap<string, number> => {
  const counts = new Map<string, number>();

  for (const section of sections) {
    if (section !== undefined) {
      counts.set(section, (counts.get(section) ?? 0) + 1);
    }
  }

  return counts;
};

/**
 * The section of each of a form's Invoke-blocking issues (undefined for one no section fixes), so each section's
 * title carries its count. For forms whose fields word their own errors; Generate's provider sets it from its issues.
 */
export const GenerationSectionIssuesProvider = ({
  children,
  issueSections,
}: {
  children: ReactNode;
  issueSections: readonly (string | undefined)[];
}) => {
  const counts = useMemo(() => countIssuesBySection(issueSections), [issueSections]);

  return <SectionIssueCountsContext value={counts}>{children}</SectionIssueCountsContext>;
};

/** The form's Invoke-blocking issues, for its sections' badges and the notes under their fields. */
export const GenerateIssuesProvider = ({
  applyFix,
  children,
  issues,
}: GenerateIssuesValue & { children: ReactNode }) => {
  const value = useMemo(() => ({ applyFix, issues }), [applyFix, issues]);
  const issueSections = useMemo(() => issues.map((issue) => issue.target.section), [issues]);

  return (
    <GenerateIssuesContext value={value}>
      <GenerationSectionIssuesProvider issueSections={issueSections}>{children}</GenerationSectionIssuesProvider>
    </GenerateIssuesContext>
  );
};

const NO_ISSUES: readonly GenerationIssue[] = [];

/** Issues with the same ids read the same, so a re-validation that finds nothing new keeps the list identity. */
export const areGenerationIssuesEqual = (
  left: readonly GenerationIssue[],
  right: readonly GenerationIssue[]
): boolean =>
  left.length === right.length &&
  left.every(
    (issue, index) =>
      issue.id === right[index]?.id &&
      issue.message === right[index]?.message &&
      JSON.stringify(issue.fix) === JSON.stringify(right[index]?.fix)
  );

/** How many issues a section holds, for its title's count. */
export const useGenerateSectionIssueCount = (section: string | undefined): number => {
  const counts = use(SectionIssueCountsContext);

  return section === undefined ? 0 : (counts?.get(section) ?? 0);
};

/**
 * Where each issue field shows its issue. Notes render under the control; the rest show the message themselves. A
 * component slot shows its own through the slot picker. A Record, so a new field cannot go without a renderer.
 */
export const ISSUE_FIELD_RENDERERS: Record<
  Exclude<GenerationIssueField, GenerateComponentValueKey>,
  'notes' | 'own message'
> = {
  cfgScale: 'notes',
  // Under the row of the concept it names, beside the row's "Not installed" badge.
  concepts: 'notes',
  krea2RebalanceWeights: 'own message',
  mossImageUncondSkipLayers: 'own message',
  // On the card of the reference image it names.
  referenceImage: 'notes',
  referenceImages: 'notes',
};

const FIX_BUTTON_SELECTOR = '[data-issue-fix]';

interface IssueTarget {
  /** Only the issues about this concept, for its row. */
  conceptKey?: string;
  field?: GenerationIssueField;
  /** Only the issues about this reference image, for its row. */
  referenceImageId?: string;
  section: GenerateSectionId;
}

const matchesIssueTarget = (
  issue: GenerationIssue,
  { conceptKey, field, referenceImageId, section }: IssueTarget
): boolean =>
  issue.target.section === section &&
  issue.target.field === field &&
  (conceptKey === undefined || (issue.fix?.kind === 'removeConcept' && issue.fix.modelKey === conceptKey)) &&
  (referenceImageId === undefined || issue.target.referenceImageId === referenceImageId);

/**
 * After a fix resolves its issue the button unmounts; focus moves to the next fix in the same notes, else to the
 * section's header, so it never falls to the page.
 */
const moveFocusAfterFix = (button: HTMLElement) => {
  const notes = button.closest('[data-issue-notes]');
  const section = button.closest('.generate-section');
  const following = notes ? [...notes.querySelectorAll<HTMLElement>(FIX_BUTTON_SELECTOR)] : [];
  const nextIndex = following.indexOf(button) + 1;

  requestAnimationFrame(() => {
    if (button.isConnected) {
      return;
    }

    const next = [...following.slice(nextIndex), ...following.slice(0, Math.max(0, nextIndex - 1))].find(
      (candidate) => candidate.isConnected
    );

    (next ?? section?.querySelector<HTMLElement>('[data-scope="collapsible"][data-part="trigger"]'))?.focus();
  });
};

/**
 * The issues a section shows itself, each with its one-step fix. Without `field` it lists the section's unplaced
 * issues. `fixesOnly` is for a control that already shows the message (the guidance scrubber): only the fix is added.
 */
export const GenerateIssueNotes = ({
  conceptKey,
  field,
  fixesOnly = false,
  referenceImageId,
  section,
}: {
  /** Only the issues about this concept, for notes under its row. */
  conceptKey?: string;
  field?: GenerationIssueField;
  fixesOnly?: boolean;
  /** Only the issues about this reference image, for notes on its card. */
  referenceImageId?: string;
  section: GenerateSectionId;
}) => {
  const { t } = useTranslation();
  const context = use(GenerateIssuesContext);
  const issues = (context?.issues ?? NO_ISSUES).filter(
    (issue) =>
      matchesIssueTarget(issue, { conceptKey, field, referenceImageId, section }) &&
      (!fixesOnly || issue.fix !== undefined)
  );

  if (!context || issues.length === 0) {
    return null;
  }

  return (
    <Stack data-issue-notes="" gap="1">
      {issues.map((issue) => {
        const message = describeGenerationIssue(t, issue);

        return (
          <HStack
            key={issue.id}
            alignItems="flex-start"
            data-issue-id={issue.id}
            gap="1.5"
            justifyContent={fixesOnly ? 'flex-end' : undefined}
          >
            {fixesOnly ? null : (
              <>
                <Icon as={CircleAlertIcon} boxSize="3" color="fg.error" flexShrink={0} mt="px" />
                <Text color="fg.error" flex="1" fontSize="xs" minW="0" textWrap="pretty">
                  {message}
                </Text>
              </>
            )}
            {issue.fix ? (
              <IssueFixButton
                fix={issue.fix}
                label={describeGenerationIssueFix(t, issue.fix)}
                name={t('widgets.generate.issueFixNamed', {
                  fix: describeGenerationIssueFix(t, issue.fix),
                  issue: message,
                })}
                onFix={context.applyFix}
              />
            ) : null}
          </HStack>
        );
      })}
    </Stack>
  );
};

const IssueFixButton = ({
  fix,
  label,
  moveFocus = moveFocusAfterFix,
  name,
  onFix,
}: {
  fix: GenerationIssueFix;
  label: string;
  /** Where focus goes once the fix has resolved the issue and removed this button. */
  moveFocus?: (button: HTMLElement) => void;
  /** The label followed by the issue it fixes, so the button names its target. */
  name: string;
  onFix: (fix: GenerationIssueFix) => void;
}) => (
  <Button
    aria-label={name}
    data-issue-fix=""
    flexShrink={0}
    size="xs"
    variant="outline"
    // oxlint-disable-next-line react-perf/jsx-no-new-function-as-prop -- one button per shown issue.
    onClick={(event) => {
      const button = event.currentTarget;

      onFix(fix);
      moveFocus(button);
    }}
  >
    {label}
  </Button>
);

/** How many issues a section holds, for sections that place their own badge. */
export const useGenerateSectionIssues = (section: GenerateSectionId): readonly GenerationIssue[] => {
  const issues = use(GenerateIssuesContext)?.issues ?? NO_ISSUES;

  return useMemo(() => issues.filter((issue) => issue.target.section === section), [issues, section]);
};

const ISSUE_POPOVER_POSITIONING = { placement: 'bottom-end' } as const;

/** A step an issue's popover offers beside its fix, for a problem fixed in the row itself (choosing its model). */
export interface GenerateIssueAction {
  label: string;
  /** Runs once the popover has closed. */
  onSelect: () => void;
}

/**
 * A list row's issues as one icon beside its switch: the row says something is wrong without a line of its own. Hover
 * or focus shows the messages; pressing it opens them in a popover with their one-step fixes. Renders nothing while
 * the row has no issue. A reveal from the Invoke issue list finds it by `data-issue-ids` and opens it.
 */
export const GenerateIssueButton = ({
  getAction,
  ...target
}: IssueTarget & {
  /** The extra step for one issue, listed ahead of its fix; null when the fix is the only way out. */
  getAction?: (issue: GenerationIssue) => GenerateIssueAction | null;
}) => {
  const { t } = useTranslation();
  const context = use(GenerateIssuesContext);
  const ids = useTooltipTriggerIds();
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const { conceptKey, field, referenceImageId, section } = target;
  const issues = useMemo(
    () =>
      (context?.issues ?? NO_ISSUES).filter((issue) =>
        matchesIssueTarget(issue, { conceptKey, field, referenceImageId, section })
      ),
    [conceptKey, context?.issues, field, referenceImageId, section]
  );
  const handleOpenChange = useCallback(({ open }: { open: boolean }) => setIsOpen(open), []);
  const close = useCallback(() => setIsOpen(false), []);
  // A fix usually resolves every issue here, and this button goes with them: focus moves to the row's switch, else to
  // the section's header, so it never falls to the page.
  const moveFocus = useCallback(() => {
    const trigger = triggerRef.current;
    const row = trigger?.closest('[role="listitem"]');
    const section = trigger?.closest('.generate-section');

    setIsOpen(false);
    requestAnimationFrame(() => {
      if (trigger?.isConnected) {
        trigger.focus();
        return;
      }

      (
        (row?.isConnected ? row.querySelector<HTMLElement>('[data-scope="switch"] input') : null) ??
        section?.querySelector<HTMLElement>('[data-scope="collapsible"][data-part="trigger"]')
      )?.focus();
    });
  }, []);

  const messages = useMemo(() => issues.map((issue) => describeGenerationIssue(t, issue)), [issues, t]);

  if (!context || issues.length === 0) {
    return null;
  }

  // The tooltip shows the name itself, so assistive tech, which drops a description equal to the name, reads it once.
  const label = t('widgets.generate.issuesNamed', { count: issues.length, messages: messages.join(' ') });

  return (
    <Popover.Root
      ids={ids}
      lazyMount
      open={isOpen}
      positioning={ISSUE_POPOVER_POSITIONING}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Tooltip content={label} ids={ids}>
        <Popover.Trigger asChild>
          <IconButton
            ref={triggerRef}
            aria-label={label}
            color="fg.error"
            data-issue-ids={issues.map((issue) => issue.id).join(' ')}
            size="sm"
            variant="ghost"
          >
            <Icon as={CircleAlertIcon} boxSize="3.5" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent maxW="18rem" p="3">
            <Stack gap="2.5">
              {issues.map((issue, index) => (
                <Stack key={issue.id} align="flex-start" data-issue-id={issue.id} gap="2">
                  <HStack align="flex-start" gap="1.5">
                    <Icon as={CircleAlertIcon} boxSize="3" color="fg.error" flexShrink={0} mt="0.5" />
                    <Text fontSize="xs" textWrap="pretty">
                      {messages[index]}
                    </Text>
                  </HStack>
                  <IssueActions
                    action={getAction?.(issue) ?? null}
                    fix={issue.fix}
                    message={messages[index] ?? ''}
                    moveFocus={moveFocus}
                    onClose={close}
                    onFix={context.applyFix}
                  />
                </Stack>
              ))}
            </Stack>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

const IssueActions = ({
  action,
  fix,
  message,
  moveFocus,
  onClose,
  onFix,
}: {
  action: GenerateIssueAction | null;
  fix: GenerationIssueFix | undefined;
  message: string;
  moveFocus: () => void;
  onClose: () => void;
  onFix: (fix: GenerationIssueFix) => void;
}) => {
  const { t } = useTranslation();
  const handleAction = useCallback(() => {
    onClose();
    action?.onSelect();
  }, [action, onClose]);

  return action || fix ? (
    <HStack gap="1.5">
      {action ? (
        <Button data-issue-action="" size="xs" variant="outline" onClick={handleAction}>
          {action.label}
        </Button>
      ) : null}
      {fix ? (
        <IssueFixButton
          fix={fix}
          label={describeGenerationIssueFix(t, fix)}
          moveFocus={moveFocus}
          name={t('widgets.generate.issueFixNamed', { fix: describeGenerationIssueFix(t, fix), issue: message })}
          onFix={onFix}
        />
      ) : null}
    </HStack>
  ) : null;
};
