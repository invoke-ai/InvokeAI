import type { LayoutPresetId, LayoutPresetSnapshot } from '@workbench/layoutContracts';
import type { AccountState } from '@workbench/projectContracts';

/**
 * Set in the `layout` of each structure the workspace model wrote to the account. A build before the split copies
 * `layout` whole wherever it keeps a structure (reading the account, applying or saving a preset), so the mark outlives
 * its rewrite of the session, which drops every account field it does not know.
 */
const WRITTEN_MARK = 'workspaceStructure';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isMarked = (snapshot: unknown): boolean =>
  isRecord(snapshot) && isRecord(snapshot.layout) && snapshot.layout[WRITTEN_MARK] === true;

/** A stored structure without the mark: the account holds which structures were written apart from them. */
export const withoutWrittenMark = <Snapshot extends LayoutPresetSnapshot>(snapshot: Snapshot): Snapshot => {
  if (!Object.hasOwn(snapshot.layout, WRITTEN_MARK)) {
    return snapshot;
  }

  const { [WRITTEN_MARK]: _mark, ...layout } = snapshot.layout as Snapshot['layout'] & Record<string, unknown>;

  return { ...snapshot, layout };
};

const withWrittenMark = (snapshot: LayoutPresetSnapshot): LayoutPresetSnapshot => ({
  ...snapshot,
  layout: { ...snapshot.layout, [WRITTEN_MARK]: true } as LayoutPresetSnapshot['layout'],
});

/** The ids, as stored, of the workspaces a stored account marks as written, and those an in-memory one lists. */
export const readWrittenStructureIds = (account: Record<string, unknown> | undefined): string[] => {
  const listed = Array.isArray(account?.writtenStructureIds)
    ? account.writtenStructureIds.filter((id): id is string => typeof id === 'string')
    : [];
  const overrides = isRecord(account?.layoutPresetOverrides) ? account.layoutPresetOverrides : {};
  const customs = Array.isArray(account?.customLayoutPresets) ? (account.customLayoutPresets as unknown[]) : [];

  return [
    ...new Set([
      ...listed,
      ...Object.entries(overrides).flatMap(([id, snapshot]) => (isMarked(snapshot) ? [id] : [])),
      ...customs.flatMap((preset) =>
        isRecord(preset) && typeof preset.id === 'string' && isMarked(preset.snapshot) ? [preset.id.trim()] : []
      ),
    ]),
  ];
};

/** The account as the session stores it: each structure the workspace model wrote marked, and the list left out. */
export const toStoredAccount = (account: AccountState): AccountState => {
  const { writtenStructureIds, ...stored } = account;

  if (!writtenStructureIds?.length) {
    return stored;
  }

  const written = new Set<LayoutPresetId>(writtenStructureIds);
  const overrides = stored.layoutPresetOverrides;

  return {
    ...stored,
    ...(stored.customLayoutPresets
      ? {
          customLayoutPresets: stored.customLayoutPresets.map((preset) =>
            written.has(preset.id) ? { ...preset, snapshot: withWrittenMark(preset.snapshot) } : preset
          ),
        }
      : {}),
    ...(overrides
      ? {
          layoutPresetOverrides: Object.fromEntries(
            Object.entries(overrides).map(([id, snapshot]) => [
              id,
              snapshot && written.has(id) ? withWrittenMark(snapshot) : snapshot,
            ])
          ),
        }
      : {}),
  };
};
