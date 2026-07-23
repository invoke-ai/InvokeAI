import type { DateTokenKey } from '@platform/search/dateTokens';
import type { TFunction } from 'i18next';

import { completeTrailingDateToken, matchTrailingDateToken } from '@platform/search/dateTokens';

import type { PaletteEntry, PaletteRow, PaletteSearchProvider, PaletteStage, ProviderResultSection } from './entries';
import type { PaletteQueryModel } from './paletteQueryModel';

import { getPaletteContributionKey } from './contributionKey';
import {
  buildProviderSectionRows,
  buildScopeRows,
  buildStageEntries,
  PROVIDER_MIN_QUERY_LENGTH,
  SEARCH_SCOPE_GROUP,
  searchPaletteRows,
} from './entries';

/**
 * Single source for stage row construction. `t` is optional so the synchronous
 * stage-preview path can build rows without translation-dependent labels;
 * rendered rows always pass it.
 */
export const buildStageRows = (
  stage: PaletteStage,
  onStageApplied: () => void,
  query: string,
  t?: TFunction
): PaletteRow[] => searchPaletteRows(buildStageEntries(stage, onStageApplied, t), query, [], { showAllOnEmpty: true });

const DATE_TOKEN_SUGGESTIONS: Record<DateTokenKey, ReadonlyArray<{ labelKey: string; value: string }>> = {
  date: [
    { labelKey: 'commandPalette.date.today', value: 'today' },
    { labelKey: 'commandPalette.date.yesterday', value: 'yesterday' },
    { labelKey: 'commandPalette.date.aWeekAgo', value: '7d' },
  ],
  from: [
    { labelKey: 'commandPalette.date.today', value: 'today' },
    { labelKey: 'commandPalette.date.yesterday', value: 'yesterday' },
    { labelKey: 'commandPalette.date.pastWeek', value: '7d' },
  ],
  to: [
    { labelKey: 'commandPalette.date.today', value: 'today' },
    { labelKey: 'commandPalette.date.yesterday', value: 'yesterday' },
  ],
};

const buildDateSuggestionRows = ({
  onCompleteDateSuggestion,
  queryModel,
  t,
}: {
  onCompleteDateSuggestion: (query: string) => void;
  queryModel: PaletteQueryModel;
  t: TFunction;
}): PaletteRow[] => {
  if (!queryModel.datesEnabled) {
    return [];
  }

  const trailing = matchTrailingDateToken(queryModel.query);

  if (!trailing) {
    return [];
  }

  const partial = trailing.partialValue.toLowerCase();
  const options = DATE_TOKEN_SUGGESTIONS[trailing.key].filter((option) => option.value.startsWith(partial));
  const rows: PaletteRow[] = [{ id: 'label:date-suggestions', kind: 'label', label: t('commandPalette.date.date') }];

  for (const option of options) {
    const id = `date-suggestion:${trailing.key}:${option.value}`;

    rows.push({
      entry: {
        group: 'Date',
        groupLabel: t('commandPalette.groups.date'),
        id,
        isPersistentRecent: false,
        keepOpen: true,
        run: () => onCompleteDateSuggestion(completeTrailingDateToken(queryModel.query, option.value)),
        subtitle: `${trailing.key}:${option.value}`,
        title: t(option.labelKey),
      },
      id,
      kind: 'entry',
    });
  }

  rows.push({ id: 'label:date-format-hint', kind: 'label', label: t('commandPalette.date.formatHint') });

  return rows;
};

const buildScopeCommandEntries = ({
  enterScope,
  providers,
  t,
}: {
  enterScope: (providerKey: string, options?: { resetQuery?: boolean }) => void;
  providers: PaletteSearchProvider[];
  t: TFunction;
}): PaletteEntry[] =>
  providers.map((provider) => ({
    group: SEARCH_SCOPE_GROUP,
    groupLabel: t('commandPalette.search.in'),
    id: getPaletteContributionKey('scope-command', provider.providerKey),
    isPersistentRecent: true,
    keepOpen: true,
    keywords: `search find browse ${provider.label.toLowerCase()}`,
    run: () => enterScope(provider.providerKey, { resetQuery: true }),
    showInEmptyState: true,
    title: t('commandPalette.search.provider', { label: provider.label.toLowerCase() }),
  }));

export const buildCommandPaletteRows = ({
  enterScope,
  entries,
  onCompleteDateSuggestion,
  onStageApplied,
  providers,
  providerSections,
  queryModel,
  recentIds,
  t,
}: {
  enterScope: (providerKey: string, options?: { resetQuery?: boolean }) => void;
  entries: PaletteEntry[];
  onCompleteDateSuggestion: (query: string) => void;
  onStageApplied: () => void;
  providers: PaletteSearchProvider[];
  providerSections: ProviderResultSection[];
  queryModel: PaletteQueryModel;
  recentIds: readonly string[];
  t: TFunction;
}): PaletteRow[] => {
  const { dateParse, isCommandsScope, isPureDateQuery, localQuery, query, scopeProvider, stage, trimmedQuery } =
    queryModel;

  if (stage) {
    return buildStageRows(stage, onStageApplied, query, t);
  }

  if (scopeProvider) {
    return buildProviderSectionRows(providerSections, null, t);
  }

  const scopeCommandEntries = buildScopeCommandEntries({ enterScope, providers, t });
  const allEntries = [...entries, ...scopeCommandEntries];
  const dateSuggestionRows = buildDateSuggestionRows({ onCompleteDateSuggestion, queryModel, t });
  const matchText = dateParse === null ? localQuery : dateParse.text;
  const hasLiveRange = dateParse?.range !== undefined;
  const localRows = isPureDateQuery
    ? []
    : searchPaletteRows(allEntries, matchText, recentIds, {
        commandsOnly: isCommandsScope,
        recentLabel: t('commandPalette.groups.recent'),
        showAllOnEmpty: isCommandsScope && trimmedQuery.length === 0,
      });

  if (isCommandsScope) {
    return localRows;
  }

  if (trimmedQuery.length === 0) {
    const syntaxHint = providers.some((provider) => provider.supportsCreatedAtRange)
      ? t('commandPalette.syntax.commandsAndDates')
      : t('commandPalette.syntax.commands');

    return [...localRows, { id: 'label:syntax-hint', kind: 'label', label: syntaxHint }];
  }

  const scopeProviders = isPureDateQuery ? providers.filter((provider) => provider.supportsCreatedAtRange) : providers;

  return [
    ...dateSuggestionRows,
    ...localRows,
    ...(matchText.trim().length >= PROVIDER_MIN_QUERY_LENGTH || hasLiveRange
      ? buildProviderSectionRows(providerSections, undefined, t)
      : []),
    ...buildScopeRows(scopeProviders, isPureDateQuery ? '' : matchText.trim(), t),
  ];
};
